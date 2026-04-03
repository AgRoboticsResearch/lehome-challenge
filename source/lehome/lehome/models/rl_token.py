import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RLTokenEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 960,
        nhead: int = 15,
        dim_feedforward: int = 1920,
        num_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.e_rl = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, z_target: torch.Tensor) -> torch.Tensor:
        B = z_target.shape[0]
        e_rl = self.e_rl.expand(B, -1, -1)
        x = torch.cat([z_target, e_rl], dim=1)
        out = self.transformer(x)
        z_rl = out[:, -1, :]
        return z_rl


class RLTokenDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 960,
        nhead: int = 15,
        dim_feedforward: int = 1920,
        num_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, z_rl: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        B, M, D = z_target.shape
        decoder_in = torch.cat(
            [z_rl.unsqueeze(1), z_target[:, :-1, :]],
            dim=1,
        )
        causal_mask = nn.Transformer.generate_square_subsequent_mask(M, device=decoder_in.device)
        output = self.transformer(decoder_in, mask=causal_mask)
        pred = self.output_proj(output)
        return pred


class RLTokenStage1(nn.Module):
    def __init__(
        self,
        d_model: int = 960,
        nhead: int = 15,
        dim_feedforward: int = 1920,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        num_image_tokens: int = 192,
        num_state_tokens: int = 1,
        num_lang_tokens: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_image_tokens = num_image_tokens
        self.num_state_tokens = num_state_tokens
        self.num_lang_tokens = num_lang_tokens

        total_prefix = num_image_tokens + num_lang_tokens + num_state_tokens
        self.register_buffer(
            "keep_mask",
            torch.cat(
                [
                    torch.ones(num_image_tokens, dtype=torch.bool),
                    torch.zeros(num_lang_tokens, dtype=torch.bool),
                    torch.ones(num_state_tokens, dtype=torch.bool),
                ]
            ),
        )

        self.encoder = RLTokenEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=encoder_layers,
        )
        self.decoder = RLTokenDecoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=decoder_layers,
        )

    def apply_keep_mask(self, z_vlm: torch.Tensor) -> torch.Tensor:
        return z_vlm[:, self.keep_mask, :]

    def forward(self, z_vlm: torch.Tensor) -> dict[str, torch.Tensor]:
        z_target = self.apply_keep_mask(z_vlm)
        z_target_sg = z_target.detach()

        z_rl = self.encoder(z_target)
        pred = self.decoder(z_rl, z_target_sg)
        loss = F.mse_loss(pred, z_target_sg)
        return {"loss": loss, "z_rl": z_rl, "pred": pred, "z_target": z_target_sg}
