# -*- coding: utf-8 -*-
"""
Script python a modo de guía de ejecucion del proyecto TFM
"""


!rm -Rf /content/tfm-oepia2
!git clone https://github.com/diegosanfuen/tfm-oepia2.git
!mkdir /content/tfm-oepia2/Logs

!chmod 775 /content/tfm-oepia2/scripts/*

!/content/tfm-oepia2/scripts/instalar_requirements.sh

!/content/tfm-oepia2/scripts/instalar_ollama.sh
!sleep 5
!/content/tfm-oepia2/scripts/instalar_ollama.sh

# Descargamos del repo
!rm -Rf /content/tfm-oepia2
!git clone https://github.com/diegosanfuen/tfm-oepia2.git
!mkdir /content/tfm-oepia2/Logs

!chmod 775 /content/tfm-oepia2/scripts/*
!nohup ollama serve &

# Ejecutar la ingesta
!/content/tfm-oepia2/scripts/ejecutar_ingesta.sh

# Descargamos del repo
!rm -Rf /content/tfm-oepia2
!git clone https://github.com/diegosanfuen/tfm-oepia2.git
!mkdir /content/tfm-oepia2/Logs

!chmod 775 /content/tfm-oepia2/scripts/*

# Ejecutar el chatbot
!nohup ollama serve &
!/content/tfm-oepia2/scripts/ejecutar_chat_opeia.sh

