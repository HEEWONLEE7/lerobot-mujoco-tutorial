import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw

class SimpleEnv:
    def __init__(self, 
                 xml_path,
                action_type='eef_pose', 
                state_type='joint_angle',
                seed = None):
        """
        args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            seed: int, seed for random number generator
        """
        # Load the xml file
        self.env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type

        self.joint_names_l = ['arm_l_joint1',
                    'arm_l_joint2',
                    'arm_l_joint3',
                    'arm_l_joint4',
                    'arm_l_joint5',
                    'arm_l_joint6',
                    'arm_l_joint7']
        
        self.joint_names_r = ['arm_r_joint1',
                    'arm_r_joint2',
                    'arm_r_joint3',
                    'arm_r_joint4',
                    'arm_r_joint5',
                    'arm_r_joint6',
                    'arm_r_joint7']
        
        self.joint_names_extra = ['lift_joint',
                    'head_joint1',
                    'head_joint2']
        
        self.init_viewer()
        self.reset(seed)

    def init_viewer(self):
        '''
        Initialize the viewer
        '''
        self.env.reset()
        self.env.init_viewer(
            distance          = 2.0,
            elevation         = -30, 
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay = False,
            loc_rgb_overlay = 'top right',
        )
    def reset(self, seed = None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        if seed != None: np.random.seed(seed=0) 
        
        # ✅ joint 이름 출력 (확인용)
        print("Left arm joints:", self.joint_names_l)
        print("Right arm joints:", self.joint_names_r)
        print("Extra joints:", self.joint_names_extra)
        
        # ✅ joint4만 -1.86으로 (ㄴ자 형태)
        q_init_l = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_init_r = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # ✅ lift는 0으로
        q_init_extra = np.array([0.0, 0.0, 0.0])  # [lift(m), head1(rad), head2(rad)]
        
        q_zero_l = q_init_l
        q_zero_r = q_init_r
        
        self.env.forward(q=q_zero_l, joint_names=self.joint_names_l, increase_tick=False)
        self.env.forward(q=q_zero_r, joint_names=self.joint_names_r, increase_tick=False)
        self.env.forward(q=q_init_extra, joint_names=self.joint_names_extra, increase_tick=False)

        # Set object positions
        obj_names = self.env.get_body_names(prefix='body_obj_')
        n_obj = len(obj_names)
        obj_xyzs = sample_xyzs(
            n_obj,
            x_range   = [+0.24,+0.4],
            y_range   = [-0.2,+0.2],
            z_range   = [0.82,0.82],
            min_dist  = 0.2,
            xy_margin = 0.0
        )
        for obj_idx in range(n_obj):
            self.env.set_p_base_body(body_name=obj_names[obj_idx],p=obj_xyzs[obj_idx,:])
            self.env.set_R_base_body(body_name=obj_names[obj_idx],R=np.eye(3,3))
        self.env.forward(increase_tick=False)

        # Set the initial pose of the robot
        self.last_q_l = copy.deepcopy(q_zero_l)
        self.last_q_r = copy.deepcopy(q_zero_r)
        self.last_q_extra = copy.deepcopy(q_init_extra)
        
        self.q_l = np.concatenate([q_zero_l, np.array([0.0]*4)])
        self.q_r = np.concatenate([q_zero_r, np.array([0.0]*4)])
        self.q_extra = q_init_extra
        
        # ✅ 초기 목표 위치를 현재 위치로 설정 (IK 계산 방지)
        self.p0_l, self.R0_l = self.env.get_pR_body(body_name='tcp_l_link')
        self.p0_r, self.R0_r = self.env.get_pR_body(body_name='tcp_r_link')
        
        mug_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_init_pose, plate_init_pose],dtype=np.float32)
        
        # ✅ step_env()만 실행 (step() 호출 안 함)
        for _ in range(100):
            self.step_env()
        
        print("✓ INITIALIZATION COMPLETE")
        self.gripper_l = False
        self.gripper_r = False

    def step(self, action_l, action_r):
        '''
        Take a step in the environment
        '''
        # ✅ 왼손: action이 있을 때만 IK 계산
        if self.action_type == 'eef_pose':
            q_l = self.env.get_qpos_joints(joint_names=self.joint_names_l)
            
            has_action_l = np.sum(np.abs(action_l[:6])) > 1e-6
            if has_action_l:
                self.p0_l += action_l[:3]
                self.R0_l = self.R0_l.dot(rpy2r(action_l[3:6]))
                
                # ✅ action이 있을 때만 IK 계산
                q_l, ik_err_stack, ik_info = solve_ik(
                    env                = self.env,
                    joint_names_for_ik = self.joint_names_l,
                    body_name_trgt     = 'tcp_l_link',
                    q_init             = q_l,
                    p_trgt             = self.p0_l,
                    R_trgt             = self.R0_l,
                    max_ik_tick        = 50,
                    ik_stepsize        = 1.0,
                    ik_eps             = 1e-2,
                    ik_th              = np.radians(5.0),
                    render             = False,
                    verbose_warning    = False,
                )
        elif self.action_type == 'delta_joint_angle':
            q_l = action_l[:-1] + self.last_q_l
        elif self.action_type == 'joint_angle':
            q_l = action_l[:-1]
        else:
            q_l = self.env.get_qpos_joints(joint_names=self.joint_names_l)
        
        # ✅ 오른손: action이 있을 때만 IK 계산
        if self.action_type == 'eef_pose':
            q_r = self.env.get_qpos_joints(joint_names=self.joint_names_r)
            
            has_action_r = np.sum(np.abs(action_r[:6])) > 1e-6
            if has_action_r:
                self.p0_r += action_r[:3]
                self.R0_r = self.R0_r.dot(rpy2r(action_r[3:6]))
                
                # ✅ action이 있을 때만 IK 계산
                q_r, ik_err_stack, ik_info = solve_ik(
                    env                = self.env,
                    joint_names_for_ik = self.joint_names_r,
                    body_name_trgt     = 'tcp_r_link',
                    q_init             = q_r,
                    p_trgt             = self.p0_r,
                    R_trgt             = self.R0_r,
                    max_ik_tick        = 50,
                    ik_stepsize        = 1.0,
                    ik_eps             = 1e-2,
                    ik_th              = np.radians(5.0),
                    render             = False,
                    verbose_warning    = False,
                )
        elif self.action_type == 'delta_joint_angle':
            q_r = action_r[:-1] + self.last_q_r
        elif self.action_type == 'joint_angle':
            q_r = action_r[:-1]
        else:
            q_r = self.env.get_qpos_joints(joint_names=self.joint_names_r)
        
        gripper_cmd_l = np.array([action_l[-1]]*4)
        gripper_cmd_l[[1,3]] *= 0.8
        gripper_cmd_r = np.array([action_r[-1]]*4)
        gripper_cmd_r[[1,3]] *= 0.8
        
        self.compute_q_l = q_l
        self.compute_q_r = q_r
        q_l = np.concatenate([q_l, gripper_cmd_l])
        q_r = np.concatenate([q_r, gripper_cmd_r])

        self.q_l = q_l
        self.q_r = q_r
        
        # ✅ Lift는 step()에서 업데이트하지 않음 (teleop_robot()에서만 제어)
        
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq_l = self.get_delta_q_l()
            dq_r = self.get_delta_q_r()
            return dq_l, dq_r

    def step_env(self):
        q_full = np.concatenate([self.q_l, self.q_r, self.q_extra])
        self.env.step(q_full)

    def grab_image(self):
        '''
        grab images from the environment
        returns:
            rgb_agent: np.array, rgb image from the agent's view
            rgb_ego: np.array, rgb image from the egocentric
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='agentview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(
            cam_name='egocentric')
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name='sideview')
        return self.rgb_agent, self.rgb_ego
        

    def plot_world_axes(self, origin=None, axis_len=0.15):
        '''
        Plot XYZ axes at the origin using spheres
        '''
        if origin is None:
            origin = np.array([0.0, 0.0, 0.82])
        
        num_points = 10
        sphere_radius = 0.004
        
        # X-axis (Red)
        for i in range(num_points):
            p = origin + np.array([axis_len * i / num_points, 0, 0])
            self.env.plot_sphere(p=p, r=sphere_radius, rgba=[1, 0, 0, 1.0])
        
        # Y-axis (Green)
        for i in range(num_points):
            p = origin + np.array([0, axis_len * i / num_points, 0])
            self.env.plot_sphere(p=p, r=sphere_radius, rgba=[0, 1, 0, 1.0])
        
        # Z-axis (Blue)
        for i in range(num_points):
            p = origin + np.array([0, 0, axis_len * i / num_points])
            self.env.plot_sphere(p=p, r=sphere_radius, rgba=[0, 0, 1, 1.0])

    def render(self, teleop=False):
        '''
        Render the environment
        '''
        self.env.plot_time()
        
        # ✅ XYZ 축 표시
        self.plot_world_axes()
        
        p_current_l, R_current_l = self.env.get_pR_body(body_name='tcp_l_link')
        p_current_r, R_current_r = self.env.get_pR_body(body_name='tcp_r_link')
        
        # 왼팔 TCP 시각화
        self.env.plot_sphere(p=p_current_l, r=0.02, rgba=[0.95,0.05,0.05,0.8])
        
        # 오른팔 TCP 시각화
        self.env.plot_sphere(p=p_current_r, r=0.02, rgba=[0.05,0.05,0.95,0.8])
        
        rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        
        self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
        self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
        self.env.render()

    def get_joint_state(self):
        '''
        Get the joint state of both arms
        returns:
            state_l, state_r: joint angles + gripper state
        '''
        qpos_l = self.env.get_qpos_joints(joint_names=self.joint_names_l)
        qpos_r = self.env.get_qpos_joints(joint_names=self.joint_names_r)
        gripper_l = self.env.get_qpos_joint('gripper_l_joint1')
        gripper_r = self.env.get_qpos_joint('gripper_r_joint1')
        gripper_cmd_l = 1.0 if gripper_l[0] > 0.5 else 0.0
        gripper_cmd_r = 1.0 if gripper_r[0] > 0.5 else 0.0
        return np.concatenate([qpos_l, [gripper_cmd_l]],dtype=np.float32), np.concatenate([qpos_r, [gripper_cmd_r]],dtype=np.float32)
    
    def teleop_robot(self):
        '''
        Teleoperate the robot using keyboard
        '''
        # 팔 전환 상태 변수 초기화
        if not hasattr(self, 'control_arm'):
            self.control_arm = 'left'
            self.gripper_l_state = False
            self.gripper_r_state = False

        # 팔 전환
        if self.env.is_key_pressed_once(key=glfw.KEY_TAB):
            self.control_arm = 'right' if self.control_arm == 'left' else 'left'
            print(f"✅ Switched to {self.control_arm.upper()} arm")

        dpos = np.zeros(3)
        drot = np.eye(3)
        d_lift = 0.0
        d_head1 = 0.0
        d_head2 = 0.0

        # ✅ ARM 움직임 (WASD + RF)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):  # 앞으로
            dpos += np.array([0.007, 0.0, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):  # 뒤로
            dpos += np.array([-0.007, 0.0, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):  # 왼쪽
            dpos += np.array([0.0, 0.007, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):  # 오른쪽
            dpos += np.array([0.0, -0.007, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):  # 위로
            dpos += np.array([0.0, 0.0, 0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):  # 아래로
            dpos += np.array([0.0, 0.0, -0.007])

        # ARM 회전 (화살표 + QE)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.03, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.03, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=0.03, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=-0.03, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=0.03, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-0.03, direction=[0.0, 1.0, 0.0])[:3, :3]

        # ✅ LIFT 제어 (V/B) - XML: range="0 0.25" (meter)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_V):
            d_lift = 0.005
        if self.env.is_key_pressed_repeat(key=glfw.KEY_B):
            d_lift = -0.005

        # ✅ HEAD 제어 (N/M + ,/.) - XML: head1: 0~0.4, head2: -0.6~0.6 (radian)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_N):
            d_head1 = 0.02
        if self.env.is_key_pressed_repeat(key=glfw.KEY_M):
            d_head1 = -0.02
        if self.env.is_key_pressed_repeat(key=glfw.KEY_COMMA):
            d_head2 = 0.02
        if self.env.is_key_pressed_repeat(key=glfw.KEY_PERIOD):
            d_head2 = -0.02

        # 리셋
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32), True

        # ✅ 그리퍼 토글 (각 팔 독립적)
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            if self.control_arm == 'left':
                self.gripper_l_state = not self.gripper_l_state
                print(f"✅ Left gripper: {'CLOSE' if self.gripper_l_state else 'OPEN'}")
            else:
                self.gripper_r_state = not self.gripper_r_state
                print(f"✅ Right gripper: {'CLOSE' if self.gripper_r_state else 'OPEN'}")

        # 액션 생성
        drot = r2rpy(drot)
        
        # ✅ 왼손 control 시 오른손 완전 고정
        if self.control_arm == 'left':
            action_l = np.concatenate([dpos, drot, np.array([self.gripper_l_state], dtype=np.float32)])
            action_r = np.zeros(8, dtype=np.float32)
            action_r[-1] = self.gripper_r_state  # 오른손 그리퍼 상태 유지
        else:
            action_l = np.zeros(8, dtype=np.float32)
            action_l[-1] = self.gripper_l_state  # 왼손 그리퍼 상태 유지
            action_r = np.concatenate([dpos, drot, np.array([self.gripper_r_state], dtype=np.float32)])

        # ✅ Lift, Head 변경 (XML 기준 Joint Limit)
        d_extra = np.array([d_lift, d_head1, d_head2], dtype=np.float32)
        self.q_extra = self.q_extra + d_extra
        # lift: 0.0~0.25(m), head1: 0~0.4(rad), head2: -0.6~0.6(rad)
        self.q_extra = np.clip(self.q_extra, 
                               [0.0, 0.0, -0.6], 
                               [0.25, 0.4, 0.6])

        return action_l, action_r, False
    
    def get_delta_q_l(self):                 
        '''
        Get the delta joint angles of the left arm
        '''
        delta = self.compute_q_l - self.last_q_l
        self.last_q_l = copy.deepcopy(self.compute_q_l)
        gripper = self.env.get_qpos_joint('gripper_l_joint1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)

    def get_delta_q_r(self):
        '''
        Get the delta joint angles of the right arm
        '''
        delta = self.compute_q_r - self.last_q_r
        self.last_q_r = copy.deepcopy(self.compute_q_r)
        gripper = self.env.get_qpos_joint('gripper_r_joint1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)

    def check_success(self):
        '''
        Check if the mug is placed on the plate
        + Gripper should be open and move upward above 0.9
        '''
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        if np.linalg.norm(p_mug[:2] - p_plate[:2]) < 0.1 and np.linalg.norm(p_mug[2] - p_plate[2]) < 0.6 and self.env.get_qpos_joint('gripper_l_joint1') < 0.1:
            p = self.env.get_p_body('tcp_l_link')[2]
            if p > 0.9:
                return True
        return False
    
    def get_obj_pose(self):
        '''
        returns: 
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        return p_mug, p_plate
    
    def set_obj_pose(self, p_mug, p_plate):
        '''
        Set the object poses
        args:
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name='body_obj_mug_5',p=p_mug)
        self.env.set_R_base_body(body_name='body_obj_mug_5',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=p_plate)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        self.step_env()


    def get_ee_pose(self):
        '''
        get the end effector pose of both arms
        returns:
            ee_pose_l, ee_pose_r: np.array, end effector pose (6,) each
        '''
        # 왼팔
        p_l, R_l = self.env.get_pR_body(body_name='tcp_l_link')
        rpy_l = r2rpy(R_l)
        ee_pose_l = np.concatenate([p_l, rpy_l], dtype=np.float32)
        
        # 오른팔
        p_r, R_r = self.env.get_pR_body(body_name='tcp_r_link')
        rpy_r = r2rpy(R_r)
        ee_pose_r = np.concatenate([p_r, rpy_r], dtype=np.float32)
        
        return ee_pose_l, ee_pose_r