


from dataclasses import dataclass
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.data.data_collator import DataCollatorForTokenClassification

@dataclass
class CustomDataCollator(DataCollatorForTokenClassification):
    def torch_call(self, features):
            import torch
            label_name = "label" if "label" in features[0].keys() else "labels"
            labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

            no_labels_features = [{k: v for k, v in feature.items() if k != label_name and k in ['input_ids', 'attention_mask', 'task_ids']} for feature in features]
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                no_labels_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            if labels is None:
                return batch

            sequence_length = batch["input_ids"].shape[1]
            padding_side = self.tokenizer.padding_side

            def to_list(tensor_or_iterable):
                if isinstance(tensor_or_iterable, torch.Tensor):
                    return tensor_or_iterable.tolist()
                return list(tensor_or_iterable)

            if padding_side == "right":
                batch[label_name] = [
                    to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
                ]
            else:
                batch[label_name] = [
                    [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
                ]

            batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
            try:
                batch['pos'] = torch.tensor([v for feature in features  for k,v in feature.items() if k == 'pos'])
            except:
                print([f['text_ids'] for f in features])
                print(batch.keys())
                print(batch['input_ids'].shape)
                #print([len(v) for feature in features  for k,v in feature.items() if k == 'pos'])
                print(len(features[-1]['pos']))
                exit()
                 
            #print(batch.keys())
            return batch