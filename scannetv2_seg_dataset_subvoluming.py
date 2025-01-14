"""
ScanNet v2 data preprocessing.
Extract point clouds data from .ply files to genrate .pickle files for training and testing.
Author: Wenxuan Wu
Date: July 2018
"""

import os
import numpy as np 
import pickle
from plyfile import PlyData

def remove_unano(scene_data, scene_label, scene_data_id):
    keep_idx = np.where((scene_label > 0) & (scene_label < 41)) # 0: unanotated
    scene_data_clean = scene_data[keep_idx]
    scene_label_clean = scene_label[keep_idx]
    scene_data_id_clean = scene_data_id[keep_idx]
    return scene_data_clean, scene_label_clean, scene_data_id_clean

test_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
def gen_label_map():
    label_map = np.zeros(41)
    for i in range(41):
        if i in test_class:
            label_map[i] = test_class.index(i)
        else:
            label_map[i] = 0
    print(label_map)
    return label_map


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= (lens + 1e-8)
    arr[:,1] /= (lens + 1e-8)
    arr[:,2] /= (lens + 1e-8)                
    return arr

def compute_normal(vertices, faces):
    #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    normals = np.zeros( vertices.shape, dtype=vertices.dtype )
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    normals[ faces[:,0] ] += n
    normals[ faces[:,1] ] += n
    normals[ faces[:,2] ] += n
    normalize_v3(normals)
    
    return normals

def gen_pickle(split = "val", root = "DataSet/Scannet_v2"):
    if split == 'test':
        root = root + "/scans_test"
    else:
        root = root + "/scans"
    file_list = "scannetv2_%s.txt"%(split)
    with open(file_list) as fl:
        scene_id = fl.read().splitlines()
    
    scene_data = []
    scene_data_labels = []
    scene_data_id = []
    scene_data_num = []
    label_map = gen_label_map()

    for i in range(len(scene_id)): #len(scene_id)
        print('process...', i)
        print('#points...', len(scene_data))
        scene_namergb = os.path.join(root, scene_id[i], scene_id[i]+'_vh_clean_2.ply')
        scene_xyzlabelrgb = PlyData.read(scene_namergb)
        scene_vertex_rgb = scene_xyzlabelrgb['vertex']
        # compute normals
        xyz = np.array([[x, y, z] for x, y, z, _, _, _, _ in scene_xyzlabelrgb["vertex"].data])
        face = np.array([f[0] for f in scene_xyzlabelrgb["face"].data])
        nxnynz = compute_normal(xyz, face)
        scene_data_tmp = np.stack((scene_vertex_rgb['x'], scene_vertex_rgb['y'],
                                   scene_vertex_rgb['z'], scene_vertex_rgb['red'],
                                   scene_vertex_rgb['green'], scene_vertex_rgb['blue'],
                                   nxnynz[:,0], nxnynz[:,1], nxnynz[:,2]), axis = -1).astype(np.float32)
        scene_points_num = scene_data_tmp.shape[0]
        scene_point_id = np.array([c for c in range(scene_points_num)])
        if split != 'test':
            scene_name = os.path.join(root, scene_id[i], scene_id[i]+'_vh_clean_2.labels.ply')
            scene_xyzlabel = PlyData.read(scene_name)
            scene_vertex = scene_xyzlabel['vertex']
            scene_data_label_tmp = scene_vertex['label']
            scene_data_tmp, scene_data_label_tmp, scene_point_id_tmp = remove_unano(scene_data_tmp, scene_data_label_tmp, scene_point_id)
        else:
            scene_data_label_tmp = np.zeros((scene_data_tmp.shape[0])).astype(np.int32)
            scene_point_id_tmp = scene_point_id
        # num = scene_data_label_tmp.shape[0]
        # idxs = np.random.choice(range(scene_data_tmp.shape[0]), 2048)
        # scene_points_num = 2048
        scene_data_label_tmp = label_map[scene_data_label_tmp]

        #subvoluming
        semantic_seg_ini = scene_data_label_tmp.astype(np.int32)
        coordmax = scene_data_tmp[:, :3].max(axis=0)
        coordmin = scene_data_tmp[:, :3].min(axis=0)
        xlength = 1.5
        ylength = 1.5
        npoints = 8192
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/xlength).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/ylength).astype(np.int32)

        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*xlength, j*ylength, 0]
                curmax = coordmin+[(i+1)*xlength, (j+1)*ylength, coordmax[2]-coordmin[2]]
                mask = np.sum((scene_data_tmp[:, :3]>=(curmin-0.01))*(scene_data_tmp[:, :3]<=(curmax+0.01)), axis=1)==3
                cur_semantic_seg = semantic_seg_ini[mask]
                if len(cur_semantic_seg) < 5000:
                    continue

                choice = np.random.choice(len(cur_semantic_seg), npoints, replace=True)

                scene_data.append(scene_data_tmp[choice])
                scene_data_labels.append(scene_data_label_tmp[choice])
                scene_data_id.append(scene_point_id_tmp[choice])
                scene_data_num.append(scene_points_num)
                # point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
                # semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN


    #scannet_train_FPS_1024.pickle
    pickle_out = open("scannet_%s_subVpoint2_normals_8192.pickle"%(split),"wb")
    pickle.dump(scene_data, pickle_out, protocol=0)
    pickle.dump(scene_data_labels, pickle_out, protocol=0)
    pickle.dump(scene_data_id, pickle_out, protocol=0)
    pickle.dump(scene_data_num, pickle_out, protocol=0)
    pickle_out.close()

if __name__ =='__main__':

    root = "/cluster/52/scannet" #modify this path to your Scannet v2 dataset Path
    gen_pickle(split = 'train', root = root)
    gen_pickle(split = 'val', root = root)
    gen_pickle(split = 'test', root = root)

    print('Done!!!')
