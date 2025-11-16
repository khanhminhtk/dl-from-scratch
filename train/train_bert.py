import torch
import torch.nn.functional as F

from model.bert import BERTModel
from model.lazylinear import LinearModel
from model.multilayer_perceptron import MultiLayerPerceptron
from optimation.optimizers import Adam


def generate_dummy_batch(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    num_classes: int,
    drive: str
):
    input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=drive)
    attention_mask = torch.zeros((batch_size, seq_length), dtype=torch.float32, device=drive)
    token_type_ids = torch.zeros_like(input_ids)

    lengths = torch.randint(low=2, high=seq_length + 1, size=(batch_size,), device=drive)
    for idx, length in enumerate(lengths):
        actual_len = int(length.item())
        actual_len = max(2, actual_len)
        body_len = max(0, actual_len - 2)
        body_tokens = torch.randint(3, vocab_size, (body_len,), device=drive)
        cls_token = torch.full((1,), 1, dtype=torch.long, device=drive)
        sep_token = torch.full((1,), 2, dtype=torch.long, device=drive)
        seq = torch.cat([cls_token, body_tokens, sep_token], dim=0)
        fill_len = min(seq_length, seq.size(0))
        input_ids[idx, :fill_len] = seq[:fill_len]
        attention_mask[idx, :fill_len] = 1.0

    labels = torch.randint(0, num_classes, (batch_size,), device=drive)
    y_true = F.one_hot(labels, num_classes=num_classes).float()
    return input_ids, token_type_ids, attention_mask, y_true, labels


def classifier_forward(classifier: LinearModel, cls_states: torch.Tensor):
    logits = classifier.fit(cls_states)
    probs = torch.softmax(logits, dim=-1)
    return logits, probs


def train_step(
    bert: BERTModel,
    classifier: LinearModel,
    optimizer: Adam,
    batch,
):
    input_ids, token_type_ids, attention_mask, y_true, true_labels = batch

    classifier.zero_grad()
    optimizer.zero_grad()

    hidden_states = bert.forward(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask
    )
    cls_states = hidden_states[:, 0, :]
    logits, probs = classifier_forward(classifier, cls_states)
    loss = MultiLayerPerceptron.cross_entropy(probs, y_true)

    batch_size = probs.shape[0]
    grad_logits = (probs - y_true) / batch_size
    classifier.grad_weights.copy_(cls_states.t().mm(grad_logits))
    classifier.grad_bias.copy_(grad_logits.sum(dim=0, keepdim=True))

    optimizer.step()
    preds = probs.argmax(dim=-1)
    acc = (preds == true_labels).float().mean().item()
    return loss.item(), acc


def main():
    drive = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    vocab_size = 32000
    seq_length = 32
    embed_dim = 64
    num_heads = 4
    hidden_dim = 128
    num_layers = 2
    num_classes = 4
    batch_size = 8
    steps = 500

    bert = BERTModel(
        vocab_size=vocab_size,
        input_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        drive=drive
    )
    classifier = LinearModel(
        input_dim=embed_dim,
        output_dim=num_classes,
        mean_init=0.0,
        std_init=0.02,
        drive=drive
    )

    optimizer = Adam(classifier.parameters(), lr=1e-2)

    for step in range(1, steps + 1):
        batch = generate_dummy_batch(batch_size, seq_length, vocab_size, num_classes, drive)
        loss, acc = train_step(bert, classifier, optimizer, batch)
        if step % 5 == 0:
            print(f"step {step:03d} | loss={loss:.4f} | acc={acc:.2%}")

    print("\nTraining finished (classifier head only — BERT weights remain frozen).")


if __name__ == "__main__":
    main()
