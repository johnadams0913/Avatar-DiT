import cv2
import torch
import numpy as np

def normalize(img, landmarks, focal_norm, distance_norm, roi_size, center, hr, ht, cam, gc=None):
    center = center.reshape(3,1)
    ## universal function for data normalization
    hR = cv2.Rodrigues(hr)[0] # rotation matrix

    ## ---------- normalize image ----------
    distance = np.linalg.norm(center) # actual distance between eye and original camera

    z_scale = distance_norm/distance
    cam_norm = np.array([
    [focal_norm, 0, roi_size[0]/2],
    [0, focal_norm, roi_size[1]/2],
    [0, 0, 1.0],
    ])
    S = np.array([ # scaling matrix
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, z_scale],
    ])

    hRx = hR[:,0]
    forward = (center/distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T # rotation matrix R
    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam))) # transformation matrix

    # if img is not None:
    # 	img_warped = cv2.warpPerspective(img, W, roi_size) # image normalization
    # else:
    # 	img_warped = None
    img_warped = cv2.warpPerspective(img, W, roi_size) # image normalization
    ## ---------- normalize rotation ----------
    hR_norm = np.dot(R, hR) # rotation matrix in normalized space
    # hr_norm = cv2.Rodrigues(hR_norm)[0] # convert rotation matrix to rotation vectors

    ## ---------- normalize gaze vector ----------
    gc_normalized = None
    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(landmarks.reshape(-1,1,2).astype('float32'), W)
    landmarks_warped = landmarks_warped.reshape(num_point, 2)
    if gc is not None:
        gc_normalized = gc.reshape((3,1)) - center # gaze vector
        # For modified data normalization, scaling is not applied to gaze direction (only R applied).
        # For original data normalization, here should be:
        # "M = np.dot(S,R)
        # gc_normalized = np.dot(R, gc_normalized)"
        gc_normalized = np.dot(R, gc_normalized)
        gc_normalized = gc_normalized/np.linalg.norm(gc_normalized)
    return [img_warped, R, hR_norm, gc_normalized, landmarks_warped, W]


def denormalize_predicted_gaze(gaze_yaw_pitch, R_inv):
	pred_gaze_cancel_nor = pitchyaw_to_vector(gaze_yaw_pitch.reshape(1,2)).reshape(3,1) # get 3d gaze direction as a vector

	pred_gaze_cancel_nor = np.matmul(R_inv, pred_gaze_cancel_nor.reshape(3,1)) # apply inverse transformation to convert it back to camera coord system
	pred_gaze_cancel_nor = pred_gaze_cancel_nor / np.linalg.norm(pred_gaze_cancel_nor) # vector normalization
	
	pred_yaw_pitch_cancel_nor = vector_to_pitchyaw(pred_gaze_cancel_nor.reshape(1,3)) # convert to yaw and pitch
	return pred_gaze_cancel_nor, pred_yaw_pitch_cancel_nor


def set_dummy_camera_model(image=None):
    assert image is not None
    h, w = image.shape[:2]
    focal_length = w*4
    center = (w//2, h//2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype = "double"
    )
    camera_distortion = np.zeros((1, 5)) # Assuming no lens distortion
    return np.array(camera_matrix), np.array(camera_distortion)


def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws: Input array of yaw and pitch angles, either numpy array or tensor.

    Returns:
        Output array of shape (n x 3) with 3D vectors per row, of the same type as the input.
    """
    if isinstance(pitchyaws, np.ndarray):
        return pitchyaw_to_vector_numpy(pitchyaws)
    elif isinstance(pitchyaws, torch.Tensor):
        return pitchyaw_to_vector_torch(pitchyaws)
    else:
        raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")


def pitchyaw_to_vector_numpy(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def pitchyaw_to_vector_torch(pitchyaws):
    n = pitchyaws.size()[0]
    sin = torch.sin(pitchyaws)
    cos = torch.cos(pitchyaws)
    out = torch.empty((n, 3), device=pitchyaws.device)
    out[:, 0] = torch.mul(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = torch.mul(cos[:, 0], cos[:, 1])
    return out


def vector_to_pitchyaw(vectors):
    """Convert given gaze vectors to pitch (theta) and yaw (phi) angles.

    Args:
        vectors: Input array of gaze vectors, either numpy array or tensor.

    Returns:
        Output array of shape (n x 2) with pitch and yaw angles, of the same type as the input.
    """
    if isinstance(vectors, np.ndarray):
        return vector_to_pitchyaw_numpy(vectors)
    elif isinstance(vectors, torch.Tensor):
        return vector_to_pitchyaw_torch(vectors)
    else:
        raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")


def vector_to_pitchyaw_numpy(vectors):
    n = vectors.shape[0]
    vectors = vectors / np.linalg.norm(vectors, axis=1).reshape(n, 1)
    out = np.empty((n, 2))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def vector_to_pitchyaw_torch(vectors):
    n = vectors.size()[0]
    vectors = vectors / torch.norm(vectors, dim=1).reshape(n, 1)
    out = torch.empty((n, 2), device=vectors.device)
    out[:, 0] = torch.asin(vectors[:, 1])  # theta
    out[:, 1] = torch.atan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
	ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

	## further optimize
	if iterate:
		ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

	return rvec, tvec


def get_face_center_by_nose(hR, ht, face_model_load):
    face_model = get_eye_nose_landmarks(face_model_load)  # the eye and nose landmarks
    Fc = np.dot(hR, face_model.T) + ht # 3D positions of facial landmarks
    face_center = mean_eye_nose(Fc.T).reshape((3, 1))  # get the face center
    return face_center, Fc


def mean_eye_nose(landmarks):
    assert landmarks.shape[0]==6
    # get the face center
    two_eye_center = np.mean(landmarks[0:4, :], axis=0).reshape(1,-1)
    nose_center = np.mean(landmarks[4:6, :], axis=0).reshape(1,-1)
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=0), axis=0).reshape(1,-1)
    return face_center


def get_eye_nose_landmarks(landmarks):
    assert landmarks.shape[0]==50 or landmarks.shape[0]==68
    if landmarks.shape[0] == 50:
        lm_6 = landmarks[[20, 23, 26, 29, 15, 19], :]  # the eye and nose landmarks
    elif landmarks.shape[0] == 68:
        lm_6 = landmarks[[36, 39, 42, 45, 31, 35], :]  # the eye and nose landmarks
    return lm_6

def angle2matrix(angles):
    device = angles.device
    x, y, z = angles[0], angles[1], angles[2]
    # x
    Rx = torch.tensor([[1, 0, 0],
                   [0, torch.cos(x), -torch.sin(x)],
                   [0, torch.sin(x), torch.cos(x)]], device=device)
    # y
    Ry = torch.tensor([[torch.cos(y), 0, torch.sin(y)],
                   [0, 1, 0],
                   [-torch.sin(y), 0, torch.cos(y)]], device=device)
    # z
    Rz = torch.tensor([[torch.cos(z), -torch.sin(z), 0],
                   [torch.sin(z), torch.cos(z), 0],
                   [0, 0, 1]], device=device)
    R = Rz @ Ry @ Rx
    return R

def angle2matrix_batch(angles):
    """
    PyTorch版本的批处理角度到旋转矩阵转换
    Args:
        angles: [bs, 3] tensor, 包含 (pitch, yaw, roll) 角度
    Returns:
        R: [bs, 3, 3] tensor, 旋转矩阵
    """
    batch_size = angles.shape[0]
    device = angles.device
    
    # 初始化旋转矩阵
    Rx = torch.zeros(batch_size, 3, 3, device=device)
    Ry = torch.zeros(batch_size, 3, 3, device=device)
    Rz = torch.zeros(batch_size, 3, 3, device=device)

    # 计算 Rx (绕X轴旋转)
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = torch.cos(angles[:, 0])
    Rx[:, 1, 2] = -torch.sin(angles[:, 0])
    Rx[:, 2, 1] = torch.sin(angles[:, 0])
    Rx[:, 2, 2] = torch.cos(angles[:, 0])

    # 计算 Ry (绕Y轴旋转)
    Ry[:, 1, 1] = 1
    Ry[:, 0, 0] = torch.cos(angles[:, 1])
    Ry[:, 0, 2] = torch.sin(angles[:, 1])
    Ry[:, 2, 0] = -torch.sin(angles[:, 1])
    Ry[:, 2, 2] = torch.cos(angles[:, 1])

    # 计算 Rz (绕Z轴旋转)
    Rz[:, 2, 2] = 1
    Rz[:, 0, 0] = torch.cos(angles[:, 2])
    Rz[:, 0, 1] = -torch.sin(angles[:, 2])
    Rz[:, 1, 0] = torch.sin(angles[:, 2])
    Rz[:, 1, 1] = torch.cos(angles[:, 2])

    # 计算 R = Rz @ Ry @ Rx
    R = torch.bmm(torch.bmm(Rz, Ry), Rx)
    return R

def vector2matrix_batch(v):
    """
    PyTorch版本的批处理向量到旋转矩阵转换
    Args:
        v: [bs, 3] tensor, 3D向量
    Returns:
        matrix: [bs, 3, 3] tensor, 旋转矩阵
    """
    angles = torch.zeros(v.shape[0], 3, device=v.device)
    angles[:, 0] = -torch.arcsin(v[:, 1])  # pitch
    angles[:, 1] = torch.arctan2(v[:, 0], v[:, 2])  # yaw
    # roll设为0，因为从向量无法确定roll角度
    
    matrix = angle2matrix_batch(angles)
    return matrix

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def batch_apply_kappa(kappa, targets, keep_length=False):
    """
    批处理版本的Kappa变换
    原理：
    R_optic * R_kappa * z = R_gaze * z
    R_optic = R_gaze * R_kappa的逆
    Args:
        kappa: [2] tensor, kappa参数
        targets: [bs, 3] tensor, 目标向量
        keep_length: bool, 是否保持长度
    Returns:
        l_optic_direction: [bs, 3] tensor, 左眼光学方向
        r_optic_direction: [bs, 3] tensor, 右眼光学方向
    """
    device = targets.device
    batch_size = targets.shape[0]
    
    # 转换kappa参数为弧度
    kappa_rad = torch.tensor(kappa, device=device) / 180 * torch.pi
    
    # 计算向量长度和单位方向
    lengths = torch.norm(targets, dim=1, keepdim=True)  # [bs, 1]
    unit_direction = targets / (lengths + 1e-8)  # [bs, 3]

    # 创建kappa旋转矩阵
    r_kappa_angles = torch.tensor([-kappa_rad[1], kappa_rad[0], 0], device=device)
    l_kappa_angles = torch.tensor([-kappa_rad[1], -kappa_rad[0], 0], device=device)
    
    r_kappa_matrix = angle2matrix(r_kappa_angles)  # [3, 3]
    l_kappa_matrix = angle2matrix(l_kappa_angles)  # [3, 3]

    # 计算逆矩阵
    r_kappa_matrix_inv = torch.inverse(r_kappa_matrix)  # [3, 3]
    l_kappa_matrix_inv = torch.inverse(l_kappa_matrix)  # [3, 3]

    # 计算gaze矩阵
    gaze_matrix = vector2matrix_batch(unit_direction)  # [bs, 3, 3]

    # 计算光学矩阵
    r_optic_matrix = torch.bmm(gaze_matrix, r_kappa_matrix_inv.unsqueeze(0).expand(batch_size, 3, 3))  # [bs, 3, 3]
    l_optic_matrix = torch.bmm(gaze_matrix, l_kappa_matrix_inv.unsqueeze(0).expand(batch_size, 3, 3))  # [bs, 3, 3]

    # 提取Z轴方向
    r_optic_direction = r_optic_matrix[:, :, 2]  # [bs, 3]
    l_optic_direction = l_optic_matrix[:, :, 2]  # [bs, 3]

    # 如果需要保持长度
    if keep_length:
        r_optic_direction = r_optic_direction * lengths
        l_optic_direction = l_optic_direction * lengths

    return l_optic_direction, r_optic_direction

def gaze_vector_to_flame_axis_angle_batch(gaze_head_coord):
    """
    批处理版本的gaze向量到FLAME轴角转换
    Args:
        gaze_head_coord: [bs, 3] tensor, 头部坐标系下的gaze向量
    Returns:
        axis_angle: [bs, 3] tensor, FLAME格式的轴角表示
    """
    batch_size = gaze_head_coord.shape[0]
    device = gaze_head_coord.device
    
    # 参考方向 (头部坐标系下的前向方向)
    reference_direction = torch.tensor([0.0, 0.0, 1.0], device=device).expand(batch_size, 3)  # [bs, 3]
    
    # 归一化
    reference_direction = reference_direction / torch.norm(reference_direction, dim=1, keepdim=True)
    gaze_head_coord = gaze_head_coord / torch.norm(gaze_head_coord, dim=1, keepdim=True)
    
    # 计算旋转轴 (叉积)
    rotation_axis = torch.cross(reference_direction, gaze_head_coord, dim=1)  # [bs, 3]
    axis_norm = torch.norm(rotation_axis, dim=1, keepdim=True)  # [bs, 1]
    
    # 避免除零
    eps = 1e-6
    rotation_axis = rotation_axis / (axis_norm + eps)
    
    # 计算旋转角度 (点积)
    cos_angle = torch.sum(reference_direction * gaze_head_coord, dim=1, keepdim=True)  # [bs, 1]
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)  # [bs, 1]
    
    # 构建轴角表示: axis * angle
    axis_angle = rotation_axis * angle  # [bs, 3]
    
    # 对于接近零向量的情况，返回零旋转
    zero_mask = axis_norm.squeeze(-1) < eps
    axis_angle[zero_mask] = 0
    
    return axis_angle


def batch_naturalize_eyemotion_code(gaze_axis_angle):
    """
    批处理版本的gaze轴角自然化
    Args:
        gaze_axis_angle: [bs, 6] tensor, 左右眼的轴角表示
    Returns:
        new_gaze_axis_angle: [bs, 6] tensor, 自然化后的轴角表示
    """
    device = gaze_axis_angle.device
    batch_size = gaze_axis_angle.shape[0]
    watch_distance = 1.5
    
    # 应用生理限制截断
    gaze_axis_angle = clamp_gaze_axis_angle_physiological(gaze_axis_angle)
    # 分离左右眼的轴角表示
    l_axis_angle, r_axis_angle = gaze_axis_angle[:, :3], gaze_axis_angle[:, 3:]
    
    # 使用批处理Rodrigues变换
    lgaze_dir = batch_rodrigues(l_axis_angle)[:, :, 2]  # [bs, 3]
    rgaze_dir = batch_rodrigues(r_axis_angle)[:, :, 2]  # [bs, 3]
    
    # 计算头部坐标系下的gaze方向
    gaze_head_coord = lgaze_dir + rgaze_dir  # [bs, 3]
    gaze_head_coord = gaze_head_coord / torch.norm(gaze_head_coord, dim=1, keepdim=True)  # [bs, 3]
    
    # 左右眼位置 [bs, 3]
    leye = torch.tensor([0.031, 0, 0], device=device).expand(batch_size, 3)  # [bs, 3]
    reye = torch.tensor([-0.031, 0, 0], device=device).expand(batch_size, 3)  # [bs, 3]
    
    # 目标位置
    target = gaze_head_coord * watch_distance  # [bs, 3]
    ltarget = target - leye  # [bs, 3]
    rtarget = target - reye  # [bs, 3]

    # 应用Kappa变换
    kappa = torch.tensor([4.0, 1.0], device=device)
    lgaze, _ = batch_apply_kappa(kappa, ltarget, keep_length=False)
    _, rgaze = batch_apply_kappa(kappa, rtarget, keep_length=False)
    
    # 归一化
    lgaze = lgaze / torch.norm(lgaze, dim=1, keepdim=True)  # [bs, 3]
    rgaze = rgaze / torch.norm(rgaze, dim=1, keepdim=True)  # [bs, 3]
    
    # 转换为FLAME轴角表示 (批处理版本)
    lgaze_axis_angle = gaze_vector_to_flame_axis_angle_batch(lgaze)  # [bs, 3]
    rgaze_axis_angle = gaze_vector_to_flame_axis_angle_batch(rgaze)  # [bs, 3]

    # 拼接结果
    new_gaze_axis_angle = torch.cat([lgaze_axis_angle, rgaze_axis_angle], dim=1)  # [bs, 6]

    return new_gaze_axis_angle

def clamp_gaze_axis_angle_physiological(gaze_axis_angle):
    """
    基于人体生理规律截断gaze轴角到合理范围
    轴角表示：第一维度=上下转动，第二维度=左右转动，第三维度=滚转
    
    根据研究，人眼的运动范围：
    - 第一维度(上下转动):约±25度
    - 第二维度(左右转动):约±45度
    - 第三维度(滚转):约±5度
    
    Args:
        gaze_axis_angle: [bs, 6] tensor, 左右眼的轴角表示
    Returns:
        clamped_gaze_axis_angle: [bs, 6] tensor, 截断后的轴角表示
    """
    
    # 分离左右眼
    l_axis_angle = gaze_axis_angle[:, :3]  # [bs, 3]
    r_axis_angle = gaze_axis_angle[:, 3:]  # [bs, 3]
    
    # 定义生理限制（弧度）
    # 这些值基于人眼实际运动范围的研究
    max_vertical_angle = 25.0 * torch.pi / 180.0    # 第一维度：上下转动 ±25度
    max_horizontal_angle = 45.0 * torch.pi / 180.0  # 第二维度：左右转动 ±45度
    max_torsion_angle = 5.0 * torch.pi / 180.0      # 第三维度：滚转 ±5度
    
    # 计算轴角表示的幅度
    l_magnitude = torch.norm(l_axis_angle, dim=1, keepdim=True)  # [bs, 1]
    r_magnitude = torch.norm(r_axis_angle, dim=1, keepdim=True)  # [bs, 1]
    
    # 避免除零
    eps = 1e-8
    l_unit = l_axis_angle / (l_magnitude + eps)  # [bs, 3]
    r_unit = r_axis_angle / (r_magnitude + eps)  # [bs, 3]
    
    # 计算各方向的投影角度
    # 轴角表示：第一维度=上下转动，第二维度=左右转动，第三维度=滚转
    l_vertical = torch.abs(l_unit[:, 0:1]) * l_magnitude    # [bs, 1] - 第一维度：上下转动
    l_horizontal = torch.abs(l_unit[:, 1:2]) * l_magnitude  # [bs, 1] - 第二维度：左右转动
    l_torsion = torch.abs(l_unit[:, 2:3]) * l_magnitude     # [bs, 1] - 第三维度：滚转
    
    r_vertical = torch.abs(r_unit[:, 0:1]) * r_magnitude    # [bs, 1] - 第一维度：上下转动
    r_horizontal = torch.abs(r_unit[:, 1:2]) * r_magnitude  # [bs, 1] - 第二维度：左右转动
    r_torsion = torch.abs(r_unit[:, 2:3]) * r_magnitude     # [bs, 1] - 第三维度：滚转
    
    # 计算缩放因子
    l_scale_h = torch.clamp(max_horizontal_angle / (l_horizontal + eps), max=1.0)  # [bs, 1]
    l_scale_v = torch.clamp(max_vertical_angle / (l_vertical + eps), max=1.0)      # [bs, 1]
    l_scale_t = torch.clamp(max_torsion_angle / (l_torsion + eps), max=1.0)        # [bs, 1]
    
    r_scale_h = torch.clamp(max_horizontal_angle / (r_horizontal + eps), max=1.0)  # [bs, 1]
    r_scale_v = torch.clamp(max_vertical_angle / (r_vertical + eps), max=1.0)      # [bs, 1]
    r_scale_t = torch.clamp(max_torsion_angle / (r_torsion + eps), max=1.0)        # [bs, 1]
    
    # 取最小缩放因子（最严格的限制）
    l_scale = torch.minimum(torch.minimum(l_scale_h, l_scale_v), l_scale_t)  # [bs, 1]
    r_scale = torch.minimum(torch.minimum(r_scale_h, r_scale_v), r_scale_t)  # [bs, 1]
    
    # 应用缩放（广播到正确的维度）
    l_clamped = l_axis_angle * l_scale  # [bs, 3] * [bs, 1] -> [bs, 3]
    r_clamped = r_axis_angle * r_scale  # [bs, 3] * [bs, 1] -> [bs, 3]
    
    # 拼接结果
    clamped_gaze_axis_angle = torch.cat([l_clamped, r_clamped], dim=1)  # [bs, 6]
    
    return clamped_gaze_axis_angle
