import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LongformerPreTrainedModel, LongformerModel


class LongformerForTokenClassification(LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.longformer = LongformerModel(config=config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]    # (last_hidden_state, hidden_states)
        sequence_output = self.dropout(sequence_output)

        # multisample dropout (wut): https://arxiv.org/abs/1905.09788
        logits1 = self.classifier(self.dropout1(sequence_output))
        logits2 = self.classifier(self.dropout2(sequence_output))
        logits3 = self.classifier(self.dropout3(sequence_output))
        logits4 = self.classifier(self.dropout4(sequence_output))
        logits5 = self.classifier(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
            loss2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
            loss3 = loss_fct(logits3.view(-1, self.num_labels), labels.view(-1))
            loss4 = loss_fct(logits4.view(-1, self.num_labels), labels.view(-1))
            loss5 = loss_fct(logits5.view(-1, self.num_labels), labels.view(-1))
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5

        logits = torch.softmax(logits, dim=-1)
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

