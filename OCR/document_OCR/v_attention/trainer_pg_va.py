from basic.generic_training_manager import GenericTrainingManager
from torch.nn import CrossEntropyLoss, CTCLoss
import torch
from basic.utils import edit_cer_from_list, edit_wer_from_list, nb_chars_from_list, nb_words_from_list, LM_ind_to_str
import numpy as np
import editdistance


class Manager(GenericTrainingManager):

    def __init__(self, params):
        super(Manager, self).__init__(params)

    def get_init_hidden(self, batch_size):
        num_layers = self.params["model_params"]["nb_layers_decoder"]
        hidden_size = self.params["model_params"]["hidden_size"]
        return torch.zeros(num_layers, batch_size, hidden_size), torch.zeros(num_layers, batch_size, hidden_size)

    def train_batch(self, batch_data, metric_names):
        loss_ctc_func = CTCLoss(blank=self.dataset.tokens["blank"], reduction="sum")
        loss_ce_func = CrossEntropyLoss(ignore_index=self.dataset.tokens["pad"])
        global_loss = 0
        total_loss_ctc = 0
        total_loss_ce = 0
        self.optimizer.zero_grad()

        x = batch_data["imgs"].to(self.device)
        y = [l.to(self.device) for l in batch_data["line_labels"]]
        y_len = batch_data["line_labels_len"]
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]

        batch_size = y[0].size()[0]

        max_nb_lines = len(y)
        for i in range(len(y), max_nb_lines):
            y.append(torch.ones((batch_size, 1)).long().to(self.device)*self.dataset.tokens["pad"])
            y_len.append([0 for _ in range(batch_size)])

        status = "init"
        features = self.models["encoder"](x)
        batch_size, c, h, w = features.size()
        attention_weights = torch.zeros((batch_size, h)).float().to(self.device)
        coverage = attention_weights.clone() if self.params["model_params"]["use_coverage_vector"] else None
        hidden = [k.to(self.device) for k in self.get_init_hidden(batch_size)] if self.params["model_params"]["use_hidden"] else None

        line_preds = [list() for _ in range(batch_size)]
        for i in range(max_nb_lines):
            context_vector, attention_weights, decision = self.models["attention"](features, attention_weights, coverage, hidden, status=status)
            coverage = coverage + attention_weights if self.params["model_params"]["use_coverage_vector"] else None
            probs, hidden = self.models["decoder"](context_vector, hidden)
            status = "inprogress"

            loss_ctc = loss_ctc_func(probs.permute(2, 0, 1), y[i], x_reduced_len, y_len[i])

            total_loss_ctc += loss_ctc.item()
            global_loss += loss_ctc
            if self.params["training_params"]["stop_mode"] == "learned":
                gt_decision = torch.ones((batch_size, )).to(self.device).long()
                for j in range(batch_size):
                    if y_len[i][j] == 0:
                        if i > 0 and y_len[i-1][j] == 0:
                            gt_decision[j] = self.dataset.tokens["pad"]
                        else:
                            gt_decision[j] = 0
                loss_ce = loss_ce_func(decision, gt_decision)
                total_loss_ce += loss_ce.item()
                global_loss += loss_ce

            line_pred = [torch.argmax(lp, dim=0).detach().cpu().numpy()[:x_reduced_len[j]] if y_len[i][j] > 0 else None for j, lp in enumerate(probs)]
            for i, lp in enumerate(line_pred):
                if lp is not None:
                    line_preds[i].append(lp)

        self.backward_loss(global_loss)
        self.optimizer.step()

        metrics = self.compute_metrics(line_preds, batch_data["raw_labels"], metric_names, from_line=True)
        if "loss_ctc" in metric_names:
            metrics["loss_ctc"] = total_loss_ctc / metrics["nb_chars"]
        if "loss_ce" in metric_names:
            metrics["loss_ce"] = total_loss_ce
        return metrics

    def evaluate_batch(self, batch_data, metric_names):

        def append_preds(pg_preds, line_preds):
            for i, lp in enumerate(line_preds):
                if lp is not None:
                    pg_preds[i].append(lp)
            return pg_preds

        x = batch_data["imgs"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]

        status = "init"
        max_nb_lines = self.params["training_params"]["max_pred_lines"]
        features = self.models["encoder"](x)
        batch_size, c, h, w = features.size()
        attention_weights = torch.zeros((batch_size, h)).float().to(self.device)
        coverage = attention_weights.clone() if self.params["model_params"]["use_coverage_vector"] else None
        hidden = [k.to(self.device) for k in self.get_init_hidden(batch_size)] if self.params["model_params"]["use_hidden"] else None
        preds = [list() for _ in range(batch_size)]
        end_pred = [None for _ in range(batch_size)]

        for i in range(max_nb_lines):
            context_vector, attention_weights, decision = self.models["attention"](features, attention_weights, coverage, hidden, status=status)
            coverage = coverage + attention_weights if self.params["model_params"]["use_coverage_vector"] else None
            probs, hidden = self.models["decoder"](context_vector, hidden)
            status = "inprogress"

            if self.params["training_params"]["stop_mode"] == "learned":
                decision = [torch.argmax(d, dim=0) for d in decision]
                for k, d in enumerate(decision):
                    if d == 0 and end_pred[k] is None:
                        end_pred[k] = i
                line_pred = [torch.argmax(lp, dim=0).detach().cpu().numpy()[:x_reduced_len[j]] if end_pred[j] is None else None for j, lp in enumerate(probs)]
                preds = append_preds(preds, line_pred)
                if np.all([end_pred[k] is not None for k in range(batch_size)]):
                    break
            else:
                line_pred = [torch.argmax(lp, dim=0).detach().cpu().numpy()[:x_reduced_len[j]] for j, lp in enumerate(probs)]
                preds = append_preds(preds, line_pred)
            ind_pred = torch.argmax(probs, dim=1)
            if torch.equal(ind_pred, torch.ones(ind_pred.size()).to(self.device).long()*self.dataset.tokens["blank"]):
                break

        metrics = self.compute_metrics(preds, batch_data["raw_labels"], metric_names, from_line=True)
        if "diff_len" in metric_names:
            end_pred = [end_pred[k] if end_pred[k] is not None else i for k in range(len(end_pred))]
            diff_len = np.array(end_pred)-np.array(batch_data["nb_lines"])
            metrics["diff_len"] = diff_len
        return metrics

    def ctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def compute_metrics(self, ind_x, str_y,  metric_names=list(), from_line=False):
        if from_line:
            str_x = list()
            for lines_token in ind_x:
                str_x.append(" ".join([LM_ind_to_str(self.dataset.charset, self.ctc_remove_successives_identical_ind(p), oov_symbol="") if p is not None else "" for p in lines_token]).strip(" "))
        else:
            str_x = [LM_ind_to_str(self.dataset.charset, self.ctc_remove_successives_identical_ind(p), oov_symbol="") if p is not None else "" for p in ind_x]
        metrics = dict()
        for metric_name in metric_names:
            if metric_name == "cer":
                metrics[metric_name] = [editdistance.eval(u, v) for u, v in zip(str_y, str_x)]
                metrics["nb_chars"] = nb_chars_from_list(str_y)
            elif metric_name == "wer":
                metrics[metric_name] = edit_wer_from_list(str_y, str_x)
                metrics["nb_words"] = nb_words_from_list(str_y)
        metrics["nb_samples"] = len(str_x)
        return metrics