import torch
model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
model.eval()
traced_model = torch.jit.trace(model, torch.randn(1,3, 224, 224))
traced_model.save('mobilenet.pt')

