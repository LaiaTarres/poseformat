import numpy as np
from tqdm import tqdm
import pdb
import imageio
import cv2
from google.protobuf.json_format import MessageToDict

from .openpose import hand_colors
from ..numpy.pose_body import NumPyPoseBody
from ..pose import Pose
from ..pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("Please install mediapipe with: pip install mediapipe")

mp_holistic = mp.solutions.holistic
mp_hands = mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BODY_POINTS = mp_holistic.PoseLandmark._member_names_
BODY_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.POSE_CONNECTIONS]

HAND_POINTS = mp_holistic.HandLandmark._member_names_
HAND_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.HAND_CONNECTIONS]

FACE_POINTS_NUM = lambda additional_points=0: additional_points + 468
FACE_POINTS = lambda additional_points=0: [str(i) for i in range(FACE_POINTS_NUM(additional_points))]
FACE_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.FACEMESH_TESSELATION]

FLIPPED_BODY_POINTS = ['NOSE', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EYE_INNER', 'LEFT_EYE',
                       'LEFT_EYE_OUTER', 'RIGHT_EAR', 'LEFT_EAR', 'MOUTH_RIGHT', 'MOUTH_LEFT', 'RIGHT_SHOULDER',
                       'LEFT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_ELBOW', 'RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_PINKY',
                       'LEFT_PINKY', 'RIGHT_INDEX', 'LEFT_INDEX', 'RIGHT_THUMB', 'LEFT_THUMB', 'RIGHT_HIP', 'LEFT_HIP',
                       'RIGHT_KNEE', 'LEFT_KNEE', 'RIGHT_ANKLE', 'LEFT_ANKLE', 'RIGHT_HEEL', 'LEFT_HEEL',
                       'RIGHT_FOOT_INDEX', 'LEFT_FOOT_INDEX', ]


def component_points(component, width: int, height: int, num: int):
    if component is not None:
        lm = component.landmark
        return np.array([[p.x * width, p.y * height, p.z] for p in lm]), np.ones(num)

    return np.zeros((num, 3)), np.zeros(num)


def body_points(component, width: int, height: int, num: int):
    if component is not None:
        lm = component.landmark
        return np.array([[p.x * width, p.y * height, p.z] for p in lm]), np.array([p.visibility for p in lm])

    return np.zeros((num, 3)), np.zeros(num)


def process_holistic(frames: list, fps: float, w: int, h: int, kinect=None, progress=False, additional_face_points=0,
                     additional_holistic_config={}):
    #Check that I don't need the model with complexity 2
    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=additional_holistic_config['min_detection_confidence'], min_tracking_confidence=additional_holistic_config['min_tracking_confidence'], model_complexity=2, smooth_landmarks=True)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=additional_holistic_config['min_detection_confidence'], min_tracking_confidence=additional_holistic_config['min_tracking_confidence'])

    #Mirar que el max num de persona sigui 1 -> Fer les hands amb mp_hands i no amb holistic
    datas = []
    confs = []

    #base = '../../../../EgoSign/visualization/examples_laia/2SnVWW3MOB4-rgb_front_how2sign'
    #path = base + '.mp4'
    #writer = imageio.get_writer(path, fps=25)

    for i, frame in enumerate(tqdm(frames, disable=not progress)):
        #results = holistic.process(frame)
        results_holistic = holistic.process(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),1))
        results_hands = hands.process(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),1)) #remove the flip, either both with flip or without

        #Get the hands the same way:
        len_results = 0
        if results_hands.multi_handedness:
            len_results = len(results_hands.multi_handedness)
        #print(f'len_results: {len_results}')
        right_hand_landmarks = None
        left_hand_landmarks = None
        right_hand_world_landmarks = None
        left_hand_world_landmarks = None
        #tant el left com el right han de ser NoneType, perquè tot funcioni bé.
        if len_results == 1: #si teneim un, només hem de mirar quin és
            handedness_dict = MessageToDict(results_hands.multi_handedness[0])
            #print(f'handedness_dict: {handedness_dict}')
            if handedness_dict['classification'][0]['label'] == 'Right':
                right_hand_landmarks = results_hands.multi_hand_landmarks[0]
                right_hand_world_landmarks = results_hands.multi_hand_world_landmarks[0]
            elif handedness_dict['classification'][0]['label'] == 'Left':
                left_hand_landmarks = results_hands.multi_hand_landmarks[0]
                left_hand_world_landmarks = results_hands.multi_hand_world_landmarks[0]
        elif len_results == 2:
            handedness_dict_0 = MessageToDict(results_hands.multi_handedness[0])
            handedness_dict_1 = MessageToDict(results_hands.multi_handedness[1])
            #print(f'handedness_dict_0: {handedness_dict_0}')
            #print(f'handedness_dict_1: {handedness_dict_1}')
            if handedness_dict_0['classification'][0]['label'] == handedness_dict_1['classification'][0]['label']: #si teneim dos iguals, quedar-nos amb el millor
                if handedness_dict_0['classification'][0]['score'] > handedness_dict_1['classification'][0]['score']:
                    if handedness_dict_0['classification'][0]['label'] == 'Right': #Veure si es right or left
                        right_hand_landmarks = results_hands.multi_hand_landmarks[0]
                        right_hand_world_landmarks = results_hands.multi_hand_world_landmarks[0]
                    elif handedness_dict_0['classification'][0]['label'] == 'Left':
                        left_hand_landmarks = results_hands.multi_hand_landmarks[0]
                        left_hand_world_landmarks = results_hands.multi_hand_world_landmarks[0]
                else: 
                    if handedness_dict_1['classification'][0]['label'] == 'Right':
                        right_hand_landmarks = results_hands.multi_hand_landmarks[1]
                        right_hand_world_landmarks = results_hands.multi_hand_world_landmarks[1]
                    elif handedness_dict_1['classification'][0]['label'] == 'Left':
                        left_hand_landmarks = results_hands.multi_hand_landmarks[1]
                        left_hand_world_landmarks = results_hands.multi_hand_world_landmarks[1]
            else: #si tenim un de cada, mirar quin és cadascun
                if handedness_dict_0['classification'][0]['label'] == 'Right':
                    right_hand_landmarks = results_hands.multi_hand_landmarks[0]
                    right_hand_world_landmarks = results_hands.multi_hand_world_landmarks[0]
                    left_hand_landmarks = results_hands.multi_hand_landmarks[1]
                    left_hand_world_landmarks = results_hands.multi_hand_world_landmarks[1]
                else:
                    right_hand_landmarks = results_hands.multi_hand_landmarks[1]
                    right_hand_world_landmarks = results_hands.multi_hand_world_landmarks[1]
                    left_hand_landmarks = results_hands.multi_hand_landmarks[0]
                    left_hand_world_landmarks = results_hands.multi_hand_world_landmarks[0]

        # Plot results in mediapipe format and save as mp4
        #mp_annotated_image = cv2.flip(frame.copy(), 1)

        #mp_drawing.draw_landmarks(mp_annotated_image, results_holistic.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        #if left_hand_landmarks:
        #    mp_drawing.draw_landmarks(mp_annotated_image, left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        #if right_hand_landmarks:
        #    mp_drawing.draw_landmarks(mp_annotated_image, right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        #writer.append_data(mp_annotated_image)

        #This is not supported in python, if I have time, implement to visualize 3d data in a video
        #mp_annotated_world_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
        #mp_drawing.draw_landmarks(mp_annotated_world_image,right_hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
        #mp_drawing.draw_landmarks(mp_annotated_world_image,left_hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
        #writer_world.append_data()

        body_data, body_confidence = body_points(results_holistic.pose_landmarks, w, h, 33)
        face_data, face_confidence = component_points(results_holistic.face_landmarks, w, h,
                                                      FACE_POINTS_NUM(additional_face_points))
        #We are changing this data for the mp_hands and adding world hands
        #lh_data, lh_confidence = component_points(results.left_hand_landmarks, w, h, 21) 
        #rh_data, rh_confidence = component_points(results.right_hand_landmarks, w, h, 21)

        lh_data, lh_confidence = component_points(left_hand_landmarks, w, h, 21) 
        rh_data, rh_confidence = component_points(right_hand_landmarks, w, h, 21)

        lh_world_data, lh_world_confidence = component_points(left_hand_world_landmarks, w, h, 21) 
        rh_world_data, rh_world_confidence = component_points(right_hand_world_landmarks, w, h, 21)

        body_world_data, body_world_confidence = body_points(results_holistic.pose_world_landmarks, w, h, 33) #this is for the body, we should have it for the hands too

        data = np.concatenate([body_data, face_data, lh_data, rh_data, body_world_data, lh_world_data, rh_world_data]) #(576 = 33 + 468 + 21 + 21 + 33) x 3
        conf = np.concatenate([body_confidence, face_confidence, lh_confidence, rh_confidence, body_world_confidence, lh_world_confidence, rh_world_confidence]) #(576 = 33 + 468 + 21 + 21 + 33) x 1

        if kinect is not None: 
            kinect_depth = []
            for x, y, z in np.array(data, dtype="int32"):
                if 0 < x < w and 0 < y < h:
                    kinect_depth.append(kinect[i, y, x, 0])
                else:
                    kinect_depth.append(0)

            kinect_vec = np.expand_dims(np.array(kinect_depth), axis=-1)
            data = np.concatenate([data, kinect_vec], axis=-1)

        datas.append(data)
        confs.append(conf)
    #writer.close()

    pose_body_data = np.expand_dims(np.stack(datas), axis=1)
    pose_body_conf = np.expand_dims(np.stack(confs), axis=1)

    holistic.close()
    hands.close()

    return NumPyPoseBody(data=pose_body_data, confidence=pose_body_conf, fps=fps)


def holistic_hand_component(name, pf="XYZC"):
    return PoseHeaderComponent(name=name, points=HAND_POINTS, limbs=HAND_LIMBS, colors=hand_colors, point_format=pf)


def holistic_components(pf="XYZC", additional_face_points=0):
    return [
        PoseHeaderComponent(name="POSE_LANDMARKS", points=BODY_POINTS, limbs=BODY_LIMBS,
                            colors=[(255, 0, 0)], point_format=pf),
        PoseHeaderComponent(name="FACE_LANDMARKS", points=FACE_POINTS(additional_face_points), limbs=FACE_LIMBS,
                            colors=[(128, 0, 0)], point_format=pf),
        holistic_hand_component("LEFT_HAND_LANDMARKS", pf),
        holistic_hand_component("RIGHT_HAND_LANDMARKS", pf),
        PoseHeaderComponent(name="POSE_WORLD_LANDMARKS", points=BODY_POINTS, limbs=BODY_LIMBS,
                            colors=[(255, 0, 0)], point_format=pf),
        holistic_hand_component("LEFT_HAND_WORLD_LANDMARKS", pf),
        holistic_hand_component("RIGHT_HAND_WORLD_LANDMARKS", pf),
    ]

def load_holistic(frames: list, fps: float = 24, width=1000, height=1000, depth=0, kinect=None, progress=False,
                  additional_holistic_config={}):
    pf = "XYZC" if kinect is None else "XYZKC"

    dimensions = PoseHeaderDimensions(width=width, height=height, depth=depth)

    refine_face_landmarks = 'refine_face_landmarks' in additional_holistic_config and additional_holistic_config[
        'refine_face_landmarks']
    additional_face_points = 10 if refine_face_landmarks else 0
    header: PoseHeader = PoseHeader(version=0.1, dimensions=dimensions,
                                    components=holistic_components(pf, additional_face_points))
    body: NumPyPoseBody = process_holistic(frames, fps, width, height, kinect, progress, additional_face_points,
                                           additional_holistic_config)

    return Pose(header, body)
