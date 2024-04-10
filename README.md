## What Is `screenshot2transcript`?
`screenshot2transcript` solves the problem of understanding the conversation given a mobile text screenshot. It can correctly 
identify dialogue bubbles and speakers (them/you). The goal is to make the model < 20 MB so we can run in on a phone. 

# Example
![image](https://github.com/kunal-mansukhani/screenshot2transcript/assets/24417742/f03aebcc-eb51-4c27-82e7-1c03dcc938e4)

![Screenshot from 2024-04-06 15-26-07](https://github.com/kunal-mansukhani/screenshot2transcript/assets/24417742/43826623-2766-4a5d-9020-7b2d7aa2528c)

Given a screenshot like this, we want to identify the speaker dialogue bubbles and sort out miscellaneous text such as timestamps, names, keyboards, etc.

# Use cases
This model will have many use cases. The most direct use case is combining it with OCR to extract the transcript of the message conversation and use it to generate smart-replies. 

# How is it done?
I'm using this paper as inspiration: https://arxiv.org/pdf/1902.08137.pdf

# Contributing
If you would like to contribute, I could use help with more training data. I currently have <50 images annotated. I use CVAT for image annotation. Also, if you could re-design the model architecture to make it smaller and maintain the level of accuracy that would be cool. 
