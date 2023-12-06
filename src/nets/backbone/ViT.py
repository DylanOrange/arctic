import timm

def vit_base_patch16_224(pretrained=True):
    model_name = "vit_base_patch16_224"
    return timm.create_model(model_name, pretrained=pretrained)