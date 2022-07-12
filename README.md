# 3d-photography-on-your-desk

Implementação da técnica descrita no artigo de 1998 "3D photography on your desk" escrito por Jean-Yves Bouguet e Pietro Perona, como trabalho final da matéria de Visão Computacional do Mestrado Strictu Sensu da Universidade Federal Fluminense
Por Horácio Macêdo e Issufi Badji

INSTRUÇÕES:

   - def.py deve estar na mesma pasta que as imagens que serão usadas como digitalização
   - (acredito que este trabalho possa ser facilmente adaptado para trabalhar com um vídeo, mas eu não testei isso)
   - toda imagem deve ser .jpg, mas isso pode ser facilmente corrigido alterando a linha 309
   - deve ter pelo menos 1 (uma) imagem camera_calibration_*
   - deve ter pelo menos 1 (uma) imagem lamp_calibration_*
   - a calibração da câmera sempre retorna uma imagem calibrada. Se a função OpenCV não funcionar como esperado, você saberá.
   - ao usar diferentes configurações de luz, deve-se alterar os pontos da imagem nas linhas 434 -- 444 seguindo o formato [y, x, z]
   - Não consegui abrir o json retornado em nenhum lugar, então não posso garantir sua utilidade.
   
***

Implementation of techniques described in the 1998 article "3D photography on your desk" by Jean-Yves Bouguet and Pietro Perona, as the final assignment for the Computer Vision class of the Strictu Sensu Master's Degree at Universidade Federal Fluminense.
By Horácio Macêdo and Issufi Badji

INSTRUCTIONS:

  - def.py must be on the same folder as the images that wil be used as a scan
  - (I believe this work can be easily adapted to work with a video, but I haven't tested that out)
  - every image must be .jpg, but this can be easily fixed by changing line 309
  - it must have at least 1 (one) camera_calibration_* image
  - it must have at least 1 (one) lamp_calibration_* image
  - camera calibration always returns a calibrated image. If the OpenCV function doesn't work as expected, you'll know.
  - while using different light setups, one must change the image points at lines 434 -- 444 following the format [y, x, z]
  - I wasn't able to open the returned json anywhere, so I can't assure you of its usefulness.
  
