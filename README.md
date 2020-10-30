# Visual Question Answering Visualizer (VQA-VIZ)
An easy-to-use app to visualise attentions of various VQA models. __Please click <a href="https://share.streamlit.io/apugoneappu/vqa_visualise/vqa_input.py" target="_blank">here</a> to see a live demo of the app!__.   



![top 7 predictions](assets/landing.jpg)

• [Models](#models)  
• [Requirements](#requirements)  
• [Installation](#installation)  
• [How to run](#how-to-run)  
• [How to use](#how-to-use)  
• [Contributing](#contributing)  
• [Acknowledgements](#acknowledgements)  

## Models

• MFB - Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering  
_Zhou Yu, Jun Yu, Jianping Fan, Dacheng Tao_  
[Arxiv](https://arxiv.org/abs/1708.01471)  

• (Coming soon) MCAN - Deep Modular Co-Attention Networks for Visual Question Answering   
_Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, Qi Tian_  
[Arvix](https://arxiv.org/abs/1906.10770)  

## Requirements
Please check the [requirements.txt](https://github.com/apugoneappu/vqa_visualise/blob/master/requirements.txt) file for the version numbers.

1. torchvision
2. seaborn
3. pandas
4. matplotlib
5. dotmap
6. streamlit
7. numpy
8. torch
9. torchvision
10. Pillow
11. PyYAML
12. opencv_python

## Installation
1. Install Anaconda 
2. Clone this repository and cd into it.  
```git clone https://github.com/apugoneappu/vqa_visualise.git && cd vqa_visualise```
3. In a new environment (`new_env`)  
```pip install -r requirements.txt```  

## How to run
From the directory of this repository, do the following -

1. ```conda activate new_env```
2. ```streamlit run vqa_input.py```
3. In a browser tab, open the Network URL displayed in your terminal.

Done! 🎉

## How to use
![input page](assets/input.jpg)
![top 7 predictions](assets/predictions.jpg)
![image attentions](assets/img_att.jpg)
![text attentions](assets/text_att.jpg)

## Contributing

First of all, thank you for wanting to contribute to this work! I will try and make your job as easy as possible.

## Acknowledgements 
This repository has been built by modifying the [OpenVQA repository](https://github.com/MILVLG/openvqa/). 

I would also like to thank [Yash Khandelwal](https://github.com/yash12khandelwal), [Nikhil Shah](https://github.com/itsshnik) and [Chinmay Singh](https://github.com/chinmay-singh) for their support and amazing suggestions!

Huge thanks to Streamlit for making all of this possible and for Streamlit Sharing that enables free hosting of this app! ❤️  

