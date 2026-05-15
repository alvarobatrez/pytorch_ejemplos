import torch

if torch.cuda.is_available():
    print(f"Dispositivo: {torch.cuda.get_device_name(0)}")
    x = torch.rand(5, 3).cuda()
    y = torch.rand(3, 4).cuda()
    z = torch.mm(x, y)
    print(f"Resultado:\n{z}")
    print(f"Ubicación del tensor: {z.device}")
else:
    print("Dispositivo: CPU")
    x = torch.rand(5, 3)
    y = torch.rand(3, 4)
    z = torch.mm(x, y)
    print(f"Resultado:\n{z}")
    print(f"Ubicación del tensor: {z.device}")
