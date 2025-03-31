# Peeling of insect embryo surface and cartography in light-sheet microscopy images


## Usage
```bash
python peel_embryo_with_cartography.py <input folder> --output_folder <output folder> --reuse_peeling --wbns_threshold mean
```

## Example outputs
![alt text](figures/embryo_surface_cartography.png)
*Cylindrical cartography projection of the surface of a peeled embryo*

## Pipeline flow

```mermaid
%%{init: { "flowchart": { "nodeSpacing": 20, "rankSpacing": 30 } } }%%
flowchart TD

    input("Load &amp; Merge Illuminations")
    maxProj1("Z-max projection") 
    thresh1("Threshold Image XY<br>to segment embryo")
    input --> maxProj1
    input --> crop("Crop Around Embryo")
    maxProj1 --> thresh1
    thresh1 --> crop
    crop --> I("Downsample &amp; Isotropize") & J("Get Isotropic Volume")
    I --> K("Peel Embryo with Cartography")
    J --> K
    K --> N
    N{"Surface Detection Mode"}
    N --> O("Surface detection based on Raycasting")
    N --> P("Surface detection<br> based on WBNS")
    O --> Q("Post-process & filter Surface Points")
    P --> Q
    Q --> qhull("Convex Hull & Binarize to mask")
    qhull --> R("Subtract Mask from Volume")
    R --> peeledVol@{ shape: procs, label: "Peeled Embryo" }
    R --> dMapping("Back project point grid to embryo & calculate Jacobian")
    dMapping --> distMap@{ shape: win-pane, label: "Cylindrical Cartography Distortion Map" }
    R --> S("Cylindrical Cartography Projection")
    S --> cartProj@{ shape: win-pane, label: "Cartography projection" }
    peeledVol --> peeledZMax@{shape: win-pane, label: "Peeled Z-max<br>projection" }
```
### Filtering for embryo structures using WBNS for embryo surface detection
![alt text](figures/embryo_structure_detection_WBNS.png)


### Erroneous signal removal from filtered embryo structures for surface detection
![alt text](figures/erroneous_signal_removal_from_WBNS_mask_with_drawing.png)

### Visualization of embryo surface points projection onto cylinder surface for cartography

![Cartography dots projected](figures/embryo_surface_points_and_proj_2.png)
![Cartography dots projected 2](figures/embryo_surface_points_and_proj.png)