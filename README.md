# grounding_dino_test

1. config 파일 다운로드 
```powershell
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py" `
  -OutFile ".\groundingdino\config\GroundingDINO_SwinB_cfg.py"
```

2. Swin-B 가중치 (.pth) 파일 다운로드
```powershell
Invoke-WebRequest `
  -Uri "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth" `
  -OutFile ".\weights\groundingdino_swinb_cogcoor.pth"
```