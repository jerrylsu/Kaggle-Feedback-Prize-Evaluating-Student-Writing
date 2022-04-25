import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import DebertaV2PreTrainedModel, DebertaV2Model


class DebertaV2ForTokenClassification(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.high_dropout = nn.Dropout(p=0.5)

        # Hidden-states of the model at the output of each layer plus the initial embedding outputs
        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[1]  # (last_hidden_state, hidden_states)
        # (4, 1600, 1024, 25) (batch_size, max_len, hidden_size, layers_num)
        sequence_outputs = torch.stack([self.dropout(layer) for layer in hidden_states], dim=3)
        # (4, 1600, 1024) (batch_size, max_len, hidden_size)
        sequence_output = (torch.softmax(self.layer_weights, dim=0) * sequence_outputs).sum(-1)

        # using last hidden layer
        # sequence_output = outputs[0]

        # multisample dropout (wut): https://arxiv.org/abs/1905.09788
        # (4, 1600, 15)  (batch_size, max_len, num_labels)
        logits = torch.mean(
            torch.stack([self.classifier(self.high_dropout(sequence_output)) for _ in range(5)], dim=0),
            dim=0,
        )
        # logits1 = self.classifier(self.dropout1(sequence_output))
        # logits2 = self.classifier(self.dropout2(sequence_output))
        # logits3 = self.classifier(self.dropout3(sequence_output))
        # logits4 = self.classifier(self.dropout4(sequence_output))
        # logits5 = self.classifier(self.dropout5(sequence_output))
        # logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # loss1 = loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
            # loss2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
            # loss3 = loss_fct(logits3.view(-1, self.num_labels), labels.view(-1))
            # loss4 = loss_fct(logits4.view(-1, self.num_labels), labels.view(-1))
            # loss5 = loss_fct(logits5.view(-1, self.num_labels), labels.view(-1))
            # loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5

        logits = torch.softmax(logits, dim=-1)
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

