# Radar_Camera_MOT
Code for our paper: **Radar-Camera Fused Multi-Object Tracking: Online Calibration and Common Feature**

## I. Abstract
This paper presents a Multi-Object Tracking (MOT) framework that fuses radar and camera data to enhance tracking efficiency while minimizing manual interventions. Contrary to many studies that underutilize radar and assign it a supplementary role—despite its capability to provide accurate range/depth information of targets in a world 3D coordinate system—our approach positions radar in a crucial role. Meanwhile, this paper utilizes common features to enable online calibration to autonomously associate detections from radar and camera. The main contributions of this work include: (1) the development of a radar-camera fusion MOT framework that exploits online radar-camera calibration to simplify the integration of detection results from these two sensors, (2) the utilization of common features between radar and camera data to accurately derive real-world positions of detected objects, and (3) the adoption of feature matching and category-consistency checking to surpass the limitations of mere position matching in enhancing sensor association accuracy. To the best of our knowledge, we are the first to investigate the integration of radar-camera common features and their use in online calibration for achieving MOT. The efficacy of our framework is demonstrated by its ability to streamline the radar-camera mapping process and improve tracking precision, as evidenced by real-world experiments conducted in both controlled environments and actual traffic scenarios.
<p align="center">
  <img src="https://github.com/radar-lab/Radar_Camera_MOT/blob/main/Figures/Fig.3.png" width="90%">
</p>

## II. Video Results
![Demo Video](https://github.com/radar-lab/Radar_Camera_MOT/blob/main/Figures/MOT_Video_RES.gif)

## III. Steps to Use this Repo
### 1. Train Common Feature Discriminator
Using **`Train_common.py`** in the folder **`Get_Common_Features\common_feats_net_car_person_final`** to train the Common Feature Discriminator model.

**`Test_common.py`** to test the accuracy of the Common Feature Discriminator.

### 2. Perform Calibration based on Common Feature
After training the Common Feature Discriminator, you can now use **`camera_to_radar_calibration.py`** or **`camera_to_radar_calibration_seperate_calib.py`** in the folder **`/Calibration_based_on_Common_Features`** to do the calibration.

Note that you should use the **`calibration.py`** in the folder **`/Calibration_based_on_Common_Features`** first to collect the point pairs between the radar and camera by using common feats, before you can do the calibration.

There are more video demos about our calibration performance, which you may refer to [our Calibration-related GitHub repository](https://github.com/radar-lab/Online-Targetless-Radar-Camera-Extrinsic-Calibration)
#### Calibration Demo video
<table>
  <tr>
    <td>
      <a href="https://www.youtube.com/watch?v=FaFU3wxIb5g">
        <img src="https://img.youtube.com/vi/FaFU3wxIb5g/0.jpg" width="460">
      </a>
    </td>
    <td>
      <a href="https://www.youtube.com/watch?v=_WVRrnrLCVU">
        <img src="https://img.youtube.com/vi/_WVRrnrLCVU/0.jpg" width="460">
      </a>
    </td>
  </tr>
</table>


### 3. Radar-Camera Fused MOT
After obtaining the calibration matrix, you can now use **`sensorfusion_tracking.py`** in the folder **`/Radar_Camera_Fused_Object_Tracking`** to do the object tracking.





## IV. High-Resolution Figures of our Paper
### Fig. 1: Visualization of Errors Caused By Pitch Angle Change

<center>
  <img src="https://github.com/radar-lab/Radar_Camera_MOT/blob/main/Figures/Fig.1.png" width=45% />
</center>

### Fig. 2: Target-based calibration scenarios typically require specific targets and environments, as well as manual intervention

<center>
  <img src="https://github.com/radar-lab/Radar_Camera_MOT/blob/main/Figures/Fig.2.png" width=45% />
</center>

### Fig. 3: Framework of The Proposed Radar-Camera Fusion MOT Method

<center>
  <img src="https://github.com/radar-lab/Radar_Camera_MOT/blob/main/Figures/Fig.3.png" width=45% />
</center>

### Fig. 4: Common Feature Discriminator Model Architecture

<center>
  <img src="https://github.com/radar-lab/Radar_Camera_MOT/blob/main/Figures/Fig.4.png" width=45% />
</center>

### Fig. 5: Example Frames Demonstrating Multi-Object Tracking Results of The Proposed Method

<center>
  <img src="https://github.com/radar-lab/Radar_Camera_MOT/blob/main/Figures/Fig.5.png" width=45% />
</center>

### Fig. 6: Test results using the Common Feature Discriminator

<center>
  <img src="https://github.com/radar-lab/Radar_Camera_MOT/blob/main/Figures/Fig.6.png" width=45% />
</center>

### Fig. 7: Homography Transformation Between Radar and Camera Planes

<center>
  <img src="https://github.com/radar-lab/Radar_Camera_MOT/blob/main/Figures/Fig.7.png" width=45% />
</center>

### Fig. 8: Block-Based Sampling Strategy

<center>
  <img src="https://github.com/radar-lab/Radar_Camera_MOT/blob/main/Figures/Fig.8.png" width=45% />
</center>

### Fig. 9: Up-Down Separation Calibration Results

<center>
  <img src="https://github.com/radar-lab/Radar_Camera_MOT/blob/main/Figures/Fig.9.png" width=45% />
</center>

### Fig. 10: Calibration Results For Three Different Scenarios

<center>
  <img src="https://github.com/radar-lab/Radar_Camera_MOT/blob/main/Figures/Fig.10.png" width=45% />
</center>

### Fig. 12: Sensor Fusion Results for Person in Scenario 3 with Camera, Radar, and Radar-then-Camera Failed, Respectively.

<center>
  <img src="https://github.com/radar-lab/Radar_Camera_MOT/blob/main/Figures/Fig.12.png" width=45% />
</center>
