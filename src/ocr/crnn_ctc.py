import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# -----------------------------
#  Charset utilities
# -----------------------------

class CharsetMapper:
    """
    Map entre caractères et indices pour plaques tunisiennes.
    On considère l'index 0 = blank CTC.
    Les tokens charset commencent à 1.
    """

    def __init__(self, charset_path: str):
        with open(charset_path, "r", encoding="utf-8") as f:
            tokens = [t.strip() for t in f.readlines() if t.strip()]

        self.tokens: List[str] = tokens
        self.blank_id: int = 0

        # mapping: token -> id (décalé de +1 pour laisser 0 = blank)
        self.token_to_id = {t: i + 1 for i, t in enumerate(tokens)}
        self.id_to_token = {i + 1: t for i, t in enumerate(tokens)}

        self.num_classes: int = len(tokens) + 1  # + blank

    def encode(self, text: str) -> List[int]:
        """
        Encode un texte de plaque en séquence d'IDs.
        On suppose que 'تونس' est un token entier du charset.
        Exemple: "121 تونس 4909"
        """
        text = text.strip()

        # on traite le mot تونس comme un token entier
        # on split grossièrement par espace
        parts = text.split()

        ids: List[int] = []
        for p in parts:
            if p == "تونس":
                ids.append(self.token_to_id["تونس"])
            else:
                # partie numérique
                for ch in p:
                    if ch in self.token_to_id:
                        ids.append(self.token_to_id[ch])
                    else:
                        # on ignore les caractères inconnus
                        pass
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Décodage (sans CTC) d'une séquence d'IDs en texte.
        On ignore les blanks et les zéros.
        """
        tokens: List[str] = []
        for i in ids:
            if i == self.blank_id:
                continue
            token = self.id_to_token.get(i, "")
            if token:
                tokens.append(token)

        # on reconstruit en séparant 'تونس' avec espaces
        out_parts: List[str] = []
        current_num = ""
        for tok in tokens:
            if tok == "تونس":
                if current_num:
                    out_parts.append(current_num)
                    current_num = ""
                out_parts.append("تونس")
            else:
                # chiffre
                current_num += tok

        if current_num:
            out_parts.append(current_num)

        return " ".join(out_parts)


# -----------------------------
#  Blocs convolutionnels optimisés
# -----------------------------

class DepthwiseSeparableConv(nn.Module):
    """
    Bloc conv optimisé type MobileNet:
    - depthwise conv
    - pointwise conv
    + BatchNorm + SiLU
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class OCRBackbone(nn.Module):
    """
    CNN léger et rapide pour extraire des features 2D.
    Entrée attendue: [B, 3, H=32, W=128] (ou proche).
    Sortie: [B, C, H', W'] avec H' petit (1 ou 2).
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()

        # [B, 3, H, W] -> [B, 32, H/2, W/2]
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # [B, 32, H/2, W/2] -> [B, 64, H/4, W/4]
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(base_ch, base_ch * 2, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # [B, 64, H/4, W/4] -> [B, 128, H/8, W/4]
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(base_ch * 2, base_ch * 4, stride=2),
        )

        # [B, 128, H/8, W/4] -> [B, 256, H/8, W/4]
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(base_ch * 4, base_ch * 8, stride=1),
            nn.Dropout2d(0.2),
        )

        self.out_channels = base_ch * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


# -----------------------------
#  CRNN + CTC
# -----------------------------

class CRNNCTC(nn.Module):
    """
    CRNN optimisé pour reconnaissance de plaques:
    - CNN léger (OCRBackbone)
    - 2 BiLSTM
    - CTC en sortie
    """

    def __init__(
        self,
        num_classes: int,
        img_h: int = 32,
        img_w: int = 128,
        in_ch: int = 3,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.25,
    ):
        """
        num_classes: nombre total de classes (y compris blank CTC)
        """
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.num_classes = num_classes

        # CNN backbone
        self.cnn = OCRBackbone(in_ch=in_ch, base_ch=32)

        # on va réduire la dimension H avec un pooling adaptatif
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, None))

        lstm_in = self.cnn.out_channels  # C
        self.bi_lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=False,  # on veut [T, B, C]
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        Retourne: log_probs [T, B, num_classes] pour CTC.
        """
        # CNN: [B, C', H', W']
        features = self.cnn(x)
        # Pool vertical -> [B, C', 1, W']
        features = self.vertical_pool(features)
        # squeeze H -> [B, C', W']
        features = features.squeeze(2)
        # permute pour RNN: [W', B, C']
        features = features.permute(2, 0, 1)

        # RNN
        seq, _ = self.bi_lstm(features)  # [T, B, 2*hidden]
        logits = self.fc(seq)            # [T, B, num_classes]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def ctc_loss(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        blank: int = 0,
    ) -> torch.Tensor:
        """
        log_probs: [T, B, C]
        targets: concaténation des labels (1D)
        target_lengths: [B]
        """
        T, B, C = log_probs.size()
        input_lengths = torch.full(
            size=(B,), fill_value=T, dtype=torch.long, device=log_probs.device
        )
        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction="mean",
            zero_infinity=True,
        )
        return loss

    @torch.no_grad()
    def greedy_decode(self, log_probs: torch.Tensor, blank: int = 0) -> List[List[int]]:
        """
        Décodage greedy simple (pour inference rapide).
        log_probs: [T, B, C]
        Retourne: liste de séquences d'IDs par batch (sans CTC collapsing).
        """
        # [T, B, C] -> [T, B]
        preds = log_probs.argmax(dim=-1)  # indices max
        preds = preds.cpu().numpy()

        sequences: List[List[int]] = []
        T, B = preds.shape
        for b in range(B):
            prev = blank
            seq: List[int] = []
            for t in range(T):
                p = int(preds[t, b])
                if p != blank and p != prev:
                    seq.append(p)
                prev = p
            sequences.append(seq)
        return sequences


# -----------------------------
#  Petit test rapide
# -----------------------------

if __name__ == "__main__":
    # test rapide du forward
    charset = CharsetMapper("src/ocr/charset_tunisian_plate.txt")
    model = CRNNCTC(num_classes=charset.num_classes)

    dummy = torch.randn(2, 3, 32, 128)  # [B, C, H, W]
    out = model(dummy)
    print("log_probs shape:", out.shape)  # [T, B, C]
