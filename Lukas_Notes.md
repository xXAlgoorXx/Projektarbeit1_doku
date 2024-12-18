# TODO
- ## Evaluierung auf PC
    - ### Model erwerb
        - Huggingface
        - Git Clip / TinyCLIP

    - ### Evaluierung
        - Code von Lia zum direkten vergleich
        - Vergleich 5 Sätze zu prompts
        - Vergleich Panoramabilder zu 5Patch

- ## Evaluierung auf Raspberry Pi ohne Hailo
    - Daten über ssh auf Rp5 kopieren `scp [source] [destination]`
    - Python script lightweight machen


- ## Evaluierung auf Raspberry Pi mit Hailo
    - ### Model erwerb
        - Huggingface
        - Git Clip / TinyCLIP
    
    - ### Model compilieren
        - DFC

- ## Metriken
    - Accuracy
    - Throughput
    - Parameter


## Notes
- Modelle von Huggingface können nicht zu ONNX umgewandelt werden wegen `scaled_dot_product_attention`, Clip vit vom original repo aber schon.
    
z_(): incompatible function arguments. The following argument types are supported:
1. (self: torch._C.Node, arg0: str, arg1: torch.Tensor) -> torch._C.Node

    Invoked with: %308 : Tensor = onnx::Constant(), scope: transformers.models.clip.modeling_clip.CLIPVisionTransformer::/transformers.models.clip.modeling_clip.CLIPEncoder::encoder/transformers.models.clip.modeling_clip.CLIPEncoderLayer::layers.0/transformers.models.clip.modeling_clip.CLIPSdpaAttention::self_attn
    , 'value', 0.125

    (Occurred when translating scaled_dot_product_attention).
    
    File "/home/lukasschoepf/Documents/Git/TinyCLIP/LukasTest/getModel.py", line 57, in <module>
    torch.onnx.export(visionModel,         # model being run
    TypeError: z_(): incompatible function arguments. The following argument types are supported:
1. (self: torch._C.Node, arg0: str, arg1: torch.Tensor) -> torch._C.Node

    Invoked with: %308 : Tensor = onnx::Constant(), scope: transformers.models.clip.modeling_clip.CLIPVisionTransformer::/transformers.models.clip.modeling_clip.CLIPEncoder::encoder/transformers.models.clip.modeling_clip.CLIPEncoderLayer::layers.0/transformers.models.clip.modeling_clip.CLIPSdpaAttention::self_attn
    , 'value', 0.125 
    (Occurred when translating scaled_dot_product_attention).


https://github.com/huggingface/transformers/issues/22221


# Latex Build

Use the following commands to build pdf with acronyms
```
pdflatex <file>
makeglossaries <file>
pdflatex <file>
pdflatex <file>
```