# VideoCallCensoring
A python app to identify faces and censor(blur/pixelate) unwanted background persons during video conferences.

# Introduction

Due to the pandemic, a lot of people have started working from home. 
And privacy has become one major concern. Unlike office environments, 
homes and other public spaces are occupied by other people who would not
prefer to be captured during video calls unintentionally. 
There are many reasons they would like to keep their identity intact. 
So, I designed a system that automatically detects & censors unknown people
walking into the camera frame during an ongoing video call. The result was a 
smooth background process that does not interrupt the subject or the members in the 
video call and runs without significant CPU load or FPS loss in the video output. 
Now, anyone can walk near a person having a video call without having to worry about their identity being revealed.

# Implementation

This is a POC implementation, so the python program starts a webcam feed to simulate a video conference app like Zoom, Google meets.

