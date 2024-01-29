import os
import torch
import csv
import numpy as np
from torch.nn import CrossEntropyLoss

from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import seed_worker
from typing import Optional

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

class CustomTrainerCallback(TrainerCallback):
    def init_csv(self, fd, csv_path):
        self.fd = fd
        self.csv_path = csv_path

    def on_train_end(self, args, state, control, **kwargs):
        print("close csv file: {}".format(self.csv_path))
        self.fd.close()

#### MFT Trainer
class LLaVATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        train_datasets = kwargs.get("train_datasets")
        if train_datasets is not None:
            max_index, max_len, max_type = max([(i, len(d), d.data_type) for i, d in enumerate(train_datasets)], key=lambda x: x[1])
            print("setting train_dataset type: {}, len: {}".format(max_type, max_len))
            self.max_index = max_index
            self.train_datasets = train_datasets
            self.custom_trainer_callback = CustomTrainerCallback()
            self.reset_dataloader_mark = {}
            kwargs["train_dataset"] = train_datasets[max_index]
            kwargs["callbacks"].append(self.custom_trainer_callback)
        else:
            self.max_index = None
            self.iterators = None
            self.train_datasets = None
        del kwargs["train_datasets"]
        super(LLaVATrainer, self).__init__(*args, **kwargs)

    ### rewrite inner training loop function, which provides MFT loss
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        if self.train_datasets is not None:
            self.init_iterators(batch_size)
            self.save_trainer_state(['epoch'] + [d.data_type for d in self.train_datasets] + ['total_loss'], init=True)
        return super(LLaVATrainer, self)._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)

    def init_iterators(self, batch_size):
        self.num_update_steps_per_epoch_map = {}
        self.dataloaders = { d.data_type: self.get_custom_dataloader(self.data_collator, d, batch_size) for index, d in enumerate(self.train_datasets) if index != self.max_index }
        self.iterators = [(iter(dataloader), data_type) for data_type, dataloader in self.dataloaders.items()]

    def get_custom_dataloader(self, data_collator, train_dataset, batch_size) -> DataLoader:
        data_collator = super(LLaVATrainer, self)._get_collator_with_removed_columns(data_collator, description="training")
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        dataloader_params["sampler"] = super(LLaVATrainer, self)._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker
        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        num_update_steps_per_epoch = len(dataloader) // self.args.gradient_accumulation_steps
        self.num_update_steps_per_epoch_map[train_dataset.data_type] = max(num_update_steps_per_epoch, 1)
        print("INFO: num_update_steps_per_epoch: {}, data_type: {}".format(num_update_steps_per_epoch, train_dataset.data_type))
        return dataloader

    #### save model and trainer state
    def save_trainer_state(self, line, init = False):
        csv_file = os.path.join(self.args.output_dir, 'trainer_state.csv')
        if init:
            self.file = open(csv_file, 'w', newline='')
            self.writer = csv.writer(self.file)
            self.custom_trainer_callback.init_csv(self.file, csv_file)
            print("save trainer state: {}, title: {}".format(csv_file, line))

        self.writer.writerow(line)
        self.file.flush()

    #### this funciton actually calculates MFT loss based on task
    def _calculate_loss(self, model_output, labels, shift_labels=False, ignore_index=-100, weighted_loss_mode='case4'):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        
        bsz, seq_len = labels.shape
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
        
        loss_mask = labels.ne(ignore_index).to(labels.dtype)

        loss_fct = CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses = losses.contiguous().view(bsz, -1)
        token_losses = losses.clone().detach().float() * loss_mask  # [B, L]

        loss = 0.0
        if weighted_loss_mode == "case3" or weighted_loss_mode == "case4":
            if weighted_loss_mode == "case3":
                loss += torch.sum((losses * loss_mask)) / torch.sum(loss_mask)
            else:
                loss += torch.mean(torch.sum(losses * loss_mask, dim=1) / torch.sum(loss_mask, dim=1))
        else:
            raise ValueError("weighted_loss_mode must be case3 or case4, others are not implemented")
        
        return loss

    #### MFT loss function
    def calculate_loss(self, model, inputs, return_outputs=False, weighted_loss_mode='case4'):
        outputs = model(**inputs)

        if "labels" in outputs:
            labels = outputs["labels"]
        else:
            labels = None
        
        if labels is None:
            raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
        
        loss = self._calculate_loss(outputs, labels, shift_labels=True, weighted_loss_mode=weighted_loss_mode)

        return (loss, outputs) if return_outputs else loss

    #### append loss for each task is MFT
    def append_loss(self, data_type, results, model, inputs, return_outputs=False, transformers=False):
        shape = inputs["input_ids"].shape
        weight = shape[0] * shape[1]
        inputs["num_dataset"] = len(self.train_datasets)
        #import pdb; pdb.set_trace()
        if return_outputs:
            #loss, outputs = super(LLaVATrainer, self).compute_loss(model, inputs, return_outputs=True)
            loss, outputs = self.calculate_loss(model, inputs, return_outputs=True, weighted_loss_mode='case3')
            results.append((loss, weight, outputs))
        else:
            #loss = super(LLaVATrainer, self).compute_loss(model, inputs, return_outputs=False)
            loss = self.calculate_loss(model, inputs, return_outputs=False, weighted_loss_mode='case3')
            results.append((loss, weight, None))
        if return_outputs is False:
            print("[{}] append_loss: {}, weight: {}, shape: {}, from transformers: {}".format(data_type, loss, weight, inputs["input_ids"].shape, transformers))

    #### rewrite trainer's compute_loss function 
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.iterators is None or return_outputs is True:
            #return super(LLaVATrainer, self).compute_loss(model, inputs, return_outputs)
            return self.calculate_loss(model, inputs, return_outputs=True, weighted_loss_mode='case3')

        results = []
        self.append_loss(self.train_datasets[self.max_index].data_type, results, model, inputs, return_outputs, True)
        for idx, (itor, data_type) in enumerate(self.iterators):
            try:
                steps_trained_in_current_epoch = self.state.global_step % self.num_update_steps_per_epoch_map[data_type]
                if self.state.global_step > 0 and steps_trained_in_current_epoch == 0:
                    reset_data_loader_key = "{}_{}".format(data_type, self.state.global_step / self.num_update_steps_per_epoch_map[data_type])
                    is_reset = self.reset_dataloader_mark.get(reset_data_loader_key, False)
                    if is_reset:
                        print("INFO: dataloader {} reseted".format(reset_data_loader_key))
                    else:
                        self.reset_dataloader_mark[reset_data_loader_key] = True
                        self.iterators[idx] = (iter(self.dataloaders[data_type]), data_type)
                        print("INFO: reset dataloader: {}, step: {}, key: {}".format(data_type, self.state.global_step, reset_data_loader_key))
                        itor = self.iterators[idx][0]
                inputs = next(itor)
                self.append_loss(data_type, results, model, inputs, return_outputs, False)
            except StopIteration:
                results.append((np.array(0), 0, None))
                print("WARNING: StopIteration: {}".format(data_type))
                continue

        # compute weighted average
        total_weight = sum([w for _, w, _ in results])
        results = [(l, w / total_weight, _) for l, w, _ in results]
        loss = sum([l * w for l, w, _ in results])
        
        print("compute_loss: {}".format(loss))
        outputs = [o for index, (_, _, o) in enumerate(results) if index == self.max_index][0]

        # save trainer state
        epoch = self.state.epoch
        self.save_trainer_state([epoch] + [l.item() for l in [l for l, _, _ in results] + [loss]])

        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
