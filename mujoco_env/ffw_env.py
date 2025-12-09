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
        q_init_l = np.deg2rad([0,0,0,0,0,0,0])
        q_init_r = np.deg2rad([0,0,0,0,0,0,0])
        q_init_extra = np.deg2rad([0,0,0])
        
        q_zero_l,ik_err_stack,ik_info = solve_ik(
            env = self.env,
            joint_names_for_ik = self.joint_names_l,
            body_name_trgt     = 'arm_base_link',
            q_init       = q_init_l,
            p_trgt       = np.array([0.3,0.0,1.0]),
            R_trgt       = rpy2r(np.deg2rad([90,-0.,90 ])),
        )
        
        q_zero_r,ik_err_stack,ik_info = solve_ik(
            env = self.env,
            joint_names_for_ik = self.joint_names_r,
            body_name_trgt     = 'arm_base_link',
            q_init       = q_init_r,
            p_trgt       = np.array([0.3,0.0,1.0]),
            R_trgt       = rpy2r(np.deg2rad([90,-0.,90 ])),
        )
        
        self.env.forward(q=q_zero_l,joint_names=self.joint_names_l,increase_tick=False)
        self.env.forward(q=q_zero_r,joint_names=self.joint_names_r,increase_tick=False)
        self.env.forward(q=q_init_extra,joint_names=self.joint_names_extra,increase_tick=False)

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
        
        self.p0_l, self.R0_l = self.env.get_pR_body(body_name='arm_base_link')
        self.p0_r, self.R0_r = self.env.get_pR_body(body_name='arm_base_link')
        
        mug_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_init_pose, plate_init_pose],dtype=np.float32)
        for _ in range(100):
            self.step_env()
        print("DONE INITIALIZATION")
        self.gripper_l = False
        self.gripper_r = False

    def step(self, action_l, action_r):
        '''
        Take a step in the environment
        args:
            action_l: np.array of shape (8,), left arm action
            action_r: np.array of shape (8,), right arm action
        returns:
            state_l, state_r: np.array, state of both arms
        '''
        if self.action_type == 'eef_pose':
            q_l = self.env.get_qpos_joints(joint_names=self.joint_names_l)
            self.p0_l += action_l[:3]
            self.R0_l = self.R0_l.dot(rpy2r(action_l[3:6]))
            q_l ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names_l,
                body_name_trgt     = 'arm_base_link',
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
        
        if self.action_type == 'eef_pose':
            q_r = self.env.get_qpos_joints(joint_names=self.joint_names_r)
            self.p0_r += action_r[:3]
            self.R0_r = self.R0_r.dot(rpy2r(action_r[3:6]))
            q_r ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names_r,
                body_name_trgt     = 'arm_base_link',
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
        

    def render(self, teleop=False):
        '''
        Render the environment
        '''
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name='arm_base_link')
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
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
        returns:
            action_l, action_r: np.array, action to take
            done: bool, True if the user wants to reset the teleoperation
        
        Keys:
            LEFT ARM (WASD + RF):
            ---------     -----------------------
               w       ->        backward
            s  a  d        left   forward   right
            ---------      -----------------------
            In x, y plane

            ---------
            R: Moving Up
            F: Moving Down
            ---------
            In z axis

            ---------
            Q: Tilt left
            E: Tilt right
            UP: Look Upward
            Down: Look Downward
            Right: Turn right
            Left: Turn left
            ---------
            For rotation

            RIGHT ARM (IJKL + UO):
            ---------
            I: backward
            K: forward
            J: left
            L: right
            U: up
            O: down
            ---------

            ---------
            z: reset
            SPACEBAR: gripper open/close
            ---------   
        '''
        dpos_l = np.zeros(3)
        dpos_r = np.zeros(3)
        drot_l = np.eye(3)
        drot_r = np.eye(3)

        # LEFT ARM CONTROL (WASD + RF)
        if self.env.is_key_pressed_repeat(glfw.KEY_S):  # +x
            dpos_l += np.array([0.007, 0, 0])
        if self.env.is_key_pressed_repeat(glfw.KEY_W):  # -x
            dpos_l += np.array([-0.007, 0, 0])
        if self.env.is_key_pressed_repeat(glfw.KEY_A):  # -y
            dpos_l += np.array([0, -0.007, 0])
        if self.env.is_key_pressed_repeat(glfw.KEY_D):  # +y
            dpos_l += np.array([0, 0.007, 0])
        if self.env.is_key_pressed_repeat(glfw.KEY_R):  # +z
            dpos_l += np.array([0, 0, 0.007])
        if self.env.is_key_pressed_repeat(glfw.KEY_F):  # -z
            dpos_l += np.array([0, 0, -0.007])

        # LEFT ARM rotation (Arrow keys + Q/E)
        if self.env.is_key_pressed_repeat(glfw.KEY_LEFT):
            drot_l = rotation_matrix(0.03, [0, 1, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(glfw.KEY_RIGHT):
            drot_l = rotation_matrix(-0.03, [0, 1, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(glfw.KEY_UP):
            drot_l = rotation_matrix(-0.03, [1, 0, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(glfw.KEY_DOWN):
            drot_l = rotation_matrix(0.03, [1, 0, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(glfw.KEY_Q):
            drot_l = rotation_matrix(0.03, [0, 0, 1])[:3, :3]
        if self.env.is_key_pressed_repeat(glfw.KEY_E):
            drot_l = rotation_matrix(-0.03, [0, 0, 1])[:3, :3]

        # LEFT gripper
        if self.env.is_key_pressed_once(glfw.KEY_SPACE):
            self.gripper_l = not getattr(self, "gripper_l", False)

        # RIGHT ARM CONTROL (IJKL + UO)
        if self.env.is_key_pressed_repeat(glfw.KEY_K):  # +x
            dpos_r += np.array([0.007, 0, 0])
        if self.env.is_key_pressed_repeat(glfw.KEY_I):  # -x
            dpos_r += np.array([-0.007, 0, 0])
        if self.env.is_key_pressed_repeat(glfw.KEY_J):  # -y
            dpos_r += np.array([0, -0.007, 0])
        if self.env.is_key_pressed_repeat(glfw.KEY_L):  # +y
            dpos_r += np.array([0, 0.007, 0])
        if self.env.is_key_pressed_repeat(glfw.KEY_U):  # +z
            dpos_r += np.array([0, 0, 0.007])
        if self.env.is_key_pressed_repeat(glfw.KEY_O):  # -z
            dpos_r += np.array([0, 0, -0.007])

        # RIGHT ARM rotation (SHIFT+Arrow + SHIFT+Q/E)
        if self.env.is_key_pressed_repeat(glfw.KEY_COMMA):  # <
            drot_r = rotation_matrix(0.03, [0, 1, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(glfw.KEY_PERIOD):  # >
            drot_r = rotation_matrix(-0.03, [0, 1, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(glfw.KEY_SEMICOLON):  # :
            drot_r = rotation_matrix(-0.03, [1, 0, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(glfw.KEY_APOSTROPHE):  # "
            drot_r = rotation_matrix(0.03, [1, 0, 0])[:3, :3]

        # RIGHT gripper
        if self.env.is_key_pressed_once(glfw.KEY_ENTER):
            self.gripper_r = not getattr(self, "gripper_r", False)

        # Reset
        if self.env.is_key_pressed_once(glfw.KEY_Z):
            return np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32), True

        drot_l = r2rpy(drot_l)
        drot_r = r2rpy(drot_r)
        action_l = np.concatenate([dpos_l, drot_l, np.array([self.gripper_l],dtype=np.float32)],dtype=np.float32)
        action_r = np.concatenate([dpos_r, drot_r, np.array([self.gripper_r],dtype=np.float32)],dtype=np.float32)
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

    def get_ee_pose(self):
        '''
        get the end effector pose of both arms
        '''
        p_l, R_l = self.env.get_pR_body(body_name='arm_base_link')
        p_r, R_r = self.env.get_pR_body(body_name='arm_base_link')
        rpy_l = r2rpy(R_l)
        rpy_r = r2rpy(R_r)
        return np.concatenate([p_l, rpy_l],dtype=np.float32), np.concatenate([p_r, rpy_r],dtype=np.float32)

    def check_success(self):
        '''
        Check if the mug is placed on the plate
        + Gripper should be open and move upward above 0.9
        '''
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        if np.linalg.norm(p_mug[:2] - p_plate[:2]) < 0.1 and np.linalg.norm(p_mug[2] - p_plate[2]) < 0.6 and self.env.get_qpos_joint('gripper_l_joint1') < 0.1:
            p = self.env.get_p_body('arm_base_link')[2]
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
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name='arm_base_link')
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)