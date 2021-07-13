import os
# import json
import numpy as np

import cv2
import torch
# from smplx import SMPL
from models.smpl import SMPL

from renderer import meshRenderer #screen less opengl renderer
from tqdm import tqdm

chang = "/media/remote_home/chang"

def visEFT_singleSubject(renderer, eft_data_all):
    smplModelPath = chang + "/datasets/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    smpl = SMPL(smplModelPath, batch_size=1, create_transl=False)

    #Visualize each EFT Fitting output
    for idx, eft_data in enumerate(tqdm(eft_data_all)):
        
        #Get raw image path
        imgFullPath = eft_data['imageName']
        # imgName = os.path.basename(imgFullPath)
        imgName = imgFullPath
        imgFullPath =os.path.join(imgDir, imgName)
        if os.path.exists(imgFullPath) ==False:
            print(f"Img path is not valid: {imgFullPath}")
            assert False
        rawImg = cv2.imread(imgFullPath)
        print(f'Input image: {imgFullPath}')

        #EFT data
        
        pred_betas = np.reshape(np.array( eft_data['parm_shape'], dtype=np.float32), (1,10) )     #(10,)
        pred_betas = torch.from_numpy(pred_betas)

        pred_pose_rotmat = np.reshape( np.array( eft_data['parm_pose'], dtype=np.float32), (1,24,3,3)  )        #(24,3,3)
        pred_pose_rotmat = torch.from_numpy(pred_pose_rotmat)


        #COCO only. Annotation index
        if 'annotId' in eft_data.keys():
            print("COCO annotId: {}".format(eft_data['annotId']))


        #Get SMPL mesh and joints from SMPL parameters
        smpl_output = smpl(betas=pred_betas, body_pose=pred_pose_rotmat[:,1:], global_orient=pred_pose_rotmat[:,[0]], pose2rot=False)
        smpl_vertices = smpl_output.vertices.detach().cpu().numpy()[0]

        
        ########################
        # Visualization
        ########################

        # Visualization Mesh
        if True: 
            pred_vert_vis = smpl_vertices
            pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}
            v = pred_meshes['ver'] 
            f = pred_meshes['f']

            #Visualize in the original image space
            renderer.set_mesh(v,f)
            renderer.showBackground(True)
            renderer.setWorldCenterBySceneCenter()
            renderer.setCameraViewMode("cam")

            #Set image size for rendering
            renderer.setViewportSize(rawImg.shape[1], rawImg.shape[0])
                
            renderer.display()
            renderImg = renderer.get_screen_color_ibgr()

        # Visualization Mesh on side view
        if True:
            renderer.showBackground(False)
            renderer.setWorldCenterBySceneCenter()
            # renderer.setCameraViewMode("side")    #To show the object in side vie
            renderer.setCameraViewMode("free")     
            renderer.setViewAngle(90,20)
            #renderer.setViewAngle(-60,50)

            #Set image size for rendering
            renderer.setViewportSize(rawImg.shape[1], rawImg.shape[0])

            renderer.display()
            sideImg = renderer.get_screen_color_ibgr()        #Overwite on rawImg
            
            sideImg = cv2.resize(sideImg, (renderImg.shape[1], renderImg.shape[0]) )
            # renderImg = cv2.resize(renderImg, (sideImg.shape[1], sideImg.shape[0]) )
        
        

        #Visualize camera view and side view
        saveImg = np.concatenate( (renderImg,sideImg), axis =1)
        # saveImg = np.concatenate( (croppedImg, renderImg,sideImg, sideImg_2), axis =1)


        #Save the rendered image to files
        if True:
            render_dir = chang + "/z_master-thesis/render_eft"
            if os.path.exists(render_dir) == False:
                os.mkdir(render_dir)
            render_output_path = render_dir + '/render_{}_eft{:08d}.jpg'.format(imgName[:-4],idx)
            print(f"Save to {render_output_path}")
            cv2.imwrite(render_output_path, saveImg)


if __name__ == '__main__':
    print("11111")
    #('--rendermode',default="geo", help="Choose among geo, normal, densepose")
    renderer = meshRenderer.meshRenderer()
    print("555")
    renderer.setRenderMode('geo')
    renderer.offscreenMode(True)
    visEFT_singleSubject(renderer)