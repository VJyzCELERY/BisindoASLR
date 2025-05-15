NUM_POSE_LANDMARKS = 19   
NUM_HAND_LANDMARKS = 21
NUM_NODES = NUM_POSE_LANDMARKS + NUM_HAND_LANDMARKS*2
FEATURE_DIM = 3 

pose_connections = [
    # Mouth
    (9,10),
    # Left Eyes
    (1,2),(2,3),(3,7),
    # Right Eyes
    (4,5),(5,6),(6,8),
    # Nose
    (0,4),(0,1),
    # Shoulders
    (11, 12),
    # Connect shoulders to hip
    (11, 17), (12, 18),
    # Connect hip points
    (17, 18),
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16)
]
hand_connections = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17)
]

pose_indices = [
    # Faces
    0,1,2,3,4,5,6,7,8,9,10,
    # Shoulders
    11, 12,
    # Arms
    13, 14, 15, 16,
    # Chest
    23, 24
]