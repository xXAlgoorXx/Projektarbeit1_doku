# TODO
- ## Evaluierung auf PC
    - ### Model erwerb
        - Huggingface
        - Git Clip/ TinyCLIP
    - ### Evaluierung
        - Code von Lia zum direkten vergleich

    - ### Metriken
        - Accuracy
        - Throughput
- ## Evaluierung auf Raspberry Pi
    - DFC Pipeline

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