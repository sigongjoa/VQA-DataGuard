import torch
import numpy as np
import os.path as osp
from smplx import SMPL as _SMPL
from smplx.utils import ModelOutput, SMPLOutput
from smplx.lbs import vertices2joints


# Map joints to SMPL joints
JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}
JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]

JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}
SMPL_MODEL_DIR = '../MotionBERT/data/mesh'
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        self.smpl_mean_params = osp.join(args[0], 'smpl_mean_params.npz')
        J_regressor_extra = np.load(osp.join(args[0], 'J_regressor_extra.npy'))
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        J_regressor_h36m = np.load(osp.join(args[0], 'J_regressor_h36m_correct.npy'))
        self.register_buffer('J_regressor_h36m', torch.tensor(J_regressor_h36m, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            joints=joints,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose)
        return output


def get_smpl_faces():
    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    return smpl.faces

import matplotlib.pyplot as plt
import numpy as np

def render_mesh_to_image_silhouette(frame, draw_face=True,
                                    elev=-90, azim=-90, roll=None,
                                    vertical_axis='z', share=False):
    """
    하나의 메쉬 프레임(frame)을 3D 렌더링 후 2D 이미지로 변환하고,
    흰색이 아닌 부분은 모두 검은색으로 처리하여 흰색 배경의 실루엣 이미지를 반환하는 함수.
    
    Args:
        frame (np.ndarray): (N, 3) 형태의 3D 메쉬 좌표 데이터.
        draw_face (bool): True이면 메쉬의 면(삼각형)을 그리며, False이면 산점도(scatter)를 사용.
        elev (float): 3D view의 elevation 각도.
        azim (float): 3D view의 azimuth 각도.
        roll (float or None): 3D view의 roll 각도.
        vertical_axis (str): 'x', 'y', 또는 'z' 중 하나로, 수직축 지정.
        share (bool): view 공유 옵션.
        
    Returns:
        silhouette_img (np.ndarray): 흰색 배경에 검은색 실루엣(2D RGB 이미지, uint8).
    """
    # 메쉬의 좌표 범위를 계산하여 동일한 축 범위를 설정
    X, Y, Z = frame[:, 0], frame[:, 1], frame[:, 2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5

    # Figure 생성
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", proj_type='ortho')
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 사용자 인자로 view 각도 설정 (roll, vertical_axis, share 포함)
    ax.view_init(elev=elev, azim=azim, roll=roll, vertical_axis=vertical_axis)
    ax.axis('off')  # 축, 글자, 타이틀 숨김

    if draw_face:
        # get_smpl_faces() 함수가 메쉬 면 정보를 반환해야 합니다.
        smpl_faces = get_smpl_faces()
        ax.plot_trisurf(frame[:, 0], frame[:, 1],
                        triangles=smpl_faces, Z=frame[:, 2],
                        color=(166/255.0, 188/255.0, 218/255.0, 0.9))
    else:
        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2],
                   s=3, c='w', edgecolors='grey')
    
    plt.tight_layout()
    # Figure를 canvas에 그린 후 2D 이미지(numpy 배열)로 변환
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rendered_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    plt.close(fig)
    
    # rendered_img에서 흰색(배경) 외의 모든 픽셀을 검은색으로 변경.
    threshold = 250  # 각 채널 값이 250 이상이면 흰색으로 간주
    mask_non_white = np.any(rendered_img < threshold, axis=-1)
    silhouette_img = np.ones_like(rendered_img) * 255  # 초기 모두 흰색
    silhouette_img[mask_non_white] = 0  # 흰색이 아닌 부분은 검은색

    return silhouette_img

