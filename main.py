import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time

# mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

W, H = 1280, 720
PUNCH_DMG = 15
KICK_DMG  = 28
PUSH_DMG  = 8
AI_MIN    = 1.2
AI_MAX    = 3.0

COLORS = {
    "player":      (100, 220, 100),
    "opponent":    (100, 100, 255),
    "hp_green":    (50,  200, 50),
    "hp_yellow":   (50,  220, 220),
    "hp_red":      (50,  50,  220),
    "white":       (255, 255, 255),
    "black":       (0,   0,   0),
    "hit_flash":   (0,   180, 255),
    "block_flash": (255, 200, 0),
    "kick_color":  (80,  80,  255),
    "punch_color": (80,  255, 80),
    "push_color":  (255, 200, 80),
}

#gesture detec
def finger_states(lm, handedness):
    tips  = [4, 8, 12, 16, 20]
    bases = [3, 6, 10, 14, 18]
    flip  = 1 if handedness == "Right" else -1
    state = [flip * (lm[4].x - lm[3].x) < 0]
    for tip, base in zip(tips[1:], bases[1:]):
        state.append(lm[tip].y < lm[base].y)
    return state

def detect_gesture(lm, handedness):
    f = finger_states(lm, handedness)
    thumb, index, middle, ring, pinky = f
    if not thumb and index and middle and not ring and not pinky:
        return "peace"
    if index and middle and ring and pinky:
        return "palm"
    if not index and not middle and not ring and not pinky:
        # thumbs up: thumb pointing upward, all fingers curled
        if lm[4].y < lm[3].y < lm[2].y:
            return "thumbs_up"
        return "fist"
    return None

#particles
class HitParticle:
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 9)
        self.x, self.y = float(x), float(y)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 2
        self.color = color
        self.life = 1.0
        self.decay = random.uniform(0.04, 0.09)
        self.size = random.randint(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.35
        self.life -= self.decay

    def draw(self, frame):
        a = self.life
        c = tuple(int(ch * a) for ch in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, c, -1)

#stickman figures

IDLE_POSE = {
    "head":       (0,   -130), "neck":       (0,   -95),
    "l_shoulder": (-30, -90),  "r_shoulder": (30,  -90),
    "l_elbow":    (-50, -55),  "r_elbow":    (50,  -55),
    "l_hand":     (-55, -20),  "r_hand":     (55,  -20),
    "l_hip":      (-18, 0),    "r_hip":      (18,  0),
    "l_knee":     (-22, 50),   "r_knee":     (22,  50),
    "l_foot":     (-26, 100),  "r_foot":     (26,  100),
}
PUNCH_POSE = {
    "head":       (0,   -130), "neck":       (0,   -95),
    "l_shoulder": (-30, -90),  "r_shoulder": (30,  -90),
    "l_elbow":    (-50, -55),  "r_elbow":    (80,  -88),
    "l_hand":     (-55, -20),  "r_hand":     (140, -88),
    "l_hip":      (-18, 0),    "r_hip":      (18,  0),
    "l_knee":     (-22, 50),   "r_knee":     (22,  50),
    "l_foot":     (-26, 100),  "r_foot":     (26,  100),
}
KICK_POSE = {
    "head":       (0,   -130), "neck":       (0,   -95),
    "l_shoulder": (-30, -90),  "r_shoulder": (30,  -90),
    "l_elbow":    (-60, -50),  "r_elbow":    (60,  -50),
    "l_hand":     (-70, -10),  "r_hand":     (70,  -10),
    "l_hip":      (-18, 0),    "r_hip":      (18,  0),
    "l_knee":     (-22, 50),   "r_knee":     (90,  -20),
    "l_foot":     (-26, 100),  "r_foot":     (160, -20),
}
PUSH_POSE = {
    "head":       (0,   -130), "neck":       (0,   -95),
    "l_shoulder": (-30, -90),  "r_shoulder": (30,  -90),
    "l_elbow":    (-80, -88),  "r_elbow":    (80,  -88),
    "l_hand":     (-130,-88),  "r_hand":     (130, -88),
    "l_hip":      (-18, 0),    "r_hip":      (18,  0),
    "l_knee":     (-30, 55),   "r_knee":     (30,  55),
    "l_foot":     (-35, 110),  "r_foot":     (35,  110),
}
BLOCK_POSE = {
    "head":       (0,   -130), "neck":       (0,   -95),
    "l_shoulder": (-30, -90),  "r_shoulder": (30,  -90),
    "l_elbow":    (-20, -115), "r_elbow":    (20,  -115),
    "l_hand":     (-30, -145), "r_hand":     (30,  -145),
    "l_hip":      (-18, 0),    "r_hip":      (18,  0),
    "l_knee":     (-22, 50),   "r_knee":     (22,  50),
    "l_foot":     (-26, 100),  "r_foot":     (26,  100),
}
HIT_POSE = {
    "head":       (20,  -120), "neck":       (15,  -88),
    "l_shoulder": (-15, -83),  "r_shoulder": (45,  -83),
    "l_elbow":    (-30, -45),  "r_elbow":    (70,  -45),
    "l_hand":     (-40, -10),  "r_hand":     (80,  -10),
    "l_hip":      (-18, 0),    "r_hip":      (18,  0),
    "l_knee":     (-22, 50),   "r_knee":     (22,  50),
    "l_foot":     (-26, 100),  "r_foot":     (26,  100),
}

class Stickman:
    def __init__(self, cx, cy, color, facing=1):
        self.cx = float(cx)
        self.cy = float(cy)
        self.color = color
        self.facing = facing
        self.hp = 100
        self.current_pose = {k: (float(v[0]), float(v[1])) for k, v in IDLE_POSE.items()}
        self.target_pose  = {k: (float(v[0]), float(v[1])) for k, v in IDLE_POSE.items()}
        self.anim_t = 1.0
        self.anim_speed = 8.0
        self.flash_timer = 0.0
        self.flash_color = COLORS["hit_flash"]
        self.knockback_vx = 0.0
        self._return_time = float("inf")

    def set_pose(self, pose_dict, speed=8.0):
        self.target_pose = {k: (float(v[0]), float(v[1])) for k, v in pose_dict.items()}
        self.anim_t = 0.0
        self.anim_speed = speed

    def flash(self, color, duration=0.25):
        self.flash_timer = duration
        self.flash_color = color

    def update(self, dt):
        # lerp pose
        self.anim_t = min(1.0, self.anim_t + dt * self.anim_speed)
        t = self.anim_t
        for key in self.target_pose:
            tx, ty = self.target_pose[key]
            cx_, cy_ = self.current_pose[key]
            self.current_pose[key] = (cx_ + (tx - cx_) * t, cy_ + (ty - cy_) * t)
        if self.flash_timer > 0:
            self.flash_timer -= dt
        self.cx += self.knockback_vx * dt * 60
        self.knockback_vx *= 0.82
        self.cx = max(80.0, min(float(W - 80), self.cx))

    def joint(self, key):
        dx, dy = self.current_pose[key]
        return (int(self.cx + dx * self.facing), int(self.cy + dy))

    def draw(self, frame):
        color = self.flash_color if self.flash_timer > 0 else self.color
        t = 4
        j = self.joint
        def line(a, b):
            cv2.line(frame, j(a), j(b), color, t, cv2.LINE_AA)
        cv2.circle(frame, j("head"), 22, color, t, cv2.LINE_AA)
        line("neck", "l_shoulder"); line("neck", "r_shoulder")
        line("neck", "l_hip");      line("neck", "r_hip")
        line("l_hip", "r_hip")
        line("l_shoulder", "l_elbow"); line("l_elbow", "l_hand")
        line("r_shoulder", "r_elbow"); line("r_elbow", "r_hand")
        line("l_hip", "l_knee");   line("l_knee", "l_foot")
        line("r_hip", "r_knee");   line("r_knee", "r_foot")

# AI opp
class AIOpponent:
    MOVES = ["fist", "peace", "palm"]
    WEIGHTS = [0.4, 0.35, 0.25]

    def __init__(self):
        self.next_move_time = time.time() + 1.5
        self.current_move = None
        self.move_end_time = 0.0
        self._last_hit_move = None

    def update(self, difficulty=1.0):
        now = time.time()
        if now >= self.next_move_time:
            self.current_move = random.choices(self.MOVES, self.WEIGHTS)[0]
            self.move_end_time = now + random.uniform(0.4, 0.9)
            interval = random.uniform(AI_MIN, AI_MAX) / difficulty
            self.next_move_time = self.move_end_time + interval
        if now >= self.move_end_time:
            self.current_move = None
        return self.current_move

# fight
def resolve_combat(attacker_move, defender_move):
    dmg_map = {"fist": PUNCH_DMG, "peace": KICK_DMG, "palm": PUSH_DMG}
    if not attacker_move:
        return False, 0, ""
    dmg = dmg_map.get(attacker_move, 0)
    if defender_move == "palm":
        if attacker_move == "peace":
            return True, dmg, "BLOCK BROKEN!"
        return False, 0, "BLOCKED!"
    return True, dmg, ""

# UI
def draw_health_bar(frame, x, y, w, h, hp, label, flip=False):
    pct = max(0, hp) / 100
    bar_color = COLORS["hp_green"] if pct > 0.5 else (COLORS["hp_yellow"] if pct > 0.25 else COLORS["hp_red"])
    cv2.rectangle(frame, (x, y), (x+w, y+h), (40,40,40), -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (80,80,80), 2)
    if flip:
        fx = x + w - int(w * pct)
        cv2.rectangle(frame, (fx, y+2), (x+w-2, y+h-2), bar_color, -1)
    else:
        cv2.rectangle(frame, (x+2, y+2), (x+2+int(w*pct)-4, y+h-2), bar_color, -1)
    cv2.putText(frame, label, (x+5, y+h-6), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLORS["white"], 2, cv2.LINE_AA)
    cv2.putText(frame, f"{max(0,int(hp))}/100", (x+5, y-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

def draw_center(frame, text, y, scale=1.5, color=(255,255,255), thick=3):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    x = (W - tw) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, COLORS["black"], thick+3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)

def draw_action_label(frame, text, x, y, color, alpha=1.0):
    c = tuple(int(ch * alpha) for ch in color)
    cv2.putText(frame, text, (int(x)-50, int(y)), cv2.FONT_HERSHEY_DUPLEX, 1.0, COLORS["black"], 4, cv2.LINE_AA)
    cv2.putText(frame, text, (int(x)-50, int(y)), cv2.FONT_HERSHEY_DUPLEX, 1.0, c, 2, cv2.LINE_AA)

def draw_legend(frame):
    bx, by = W-260, H-100
    cv2.rectangle(frame, (bx-10, by-20), (W-10, H-10), (20,20,20), -1)
    cv2.rectangle(frame, (bx-10, by-20), (W-10, H-10), (60,60,60), 1)
    items = [
        ("Fist  -> Punch  15hp", COLORS["punch_color"]),
        ("Peace -> Kick   28hp", COLORS["kick_color"]),
        ("Palm  -> Block/Push", COLORS["push_color"]),
    ]
    for i, (txt, col) in enumerate(items):
        cv2.putText(frame, txt, (bx, by + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

# main
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    # --- state ---
    player   = Stickman(W//4,     H-80, COLORS["player"],   facing=1)
    opponent = Stickman(W*3//4,   H-80, COLORS["opponent"], facing=-1)
    ai = AIOpponent()

    player_wins = 0
    ai_wins     = 0
    round_num   = 1
    difficulty  = 1.0
    state       = "waiting"   # waiting, fighting, round_end, game_over
    round_end_timer = 0.0
    winner_text = ""

    particles    = []
    action_labels = []   
    screen_shake = 0.0

    cur_gesture  = None
    gesture_hold = 0
    HOLD_THRESH  = 5
    stable_gesture = None
    player_action_time = 0.0
    ACTION_COOLDOWN    = 0.65

    prev_time = time.time()
    print("Stickman Fight ready — show THUMBS UP to start | Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()
        dt  = min(now - prev_time, 0.05)
        prev_time = now

        # screen shake offset
        sdx = sdy = 0
        if screen_shake > 0:
            mag = int(screen_shake * 14)
            sdx = random.randint(-mag, mag)
            sdy = random.randint(-mag, mag)

        frame = (frame * 0.5).astype(np.uint8)

        # hand detection
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        raw_gesture = None
        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_lm, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                raw_gesture = detect_gesture(hand_lm.landmark, hand_info.classification[0].label)
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(180,220,255), thickness=1))

        # gesture stabiliser
        if raw_gesture == cur_gesture:
            gesture_hold = min(gesture_hold+1, HOLD_THRESH+1)
        else:
            cur_gesture  = raw_gesture
            gesture_hold = 0
        if gesture_hold >= HOLD_THRESH:
            stable_gesture = cur_gesture
        elif raw_gesture is None:
            stable_gesture = None

        # state machine
        if state == "waiting":
            player.set_pose(IDLE_POSE, speed=4)
            opponent.set_pose(IDLE_POSE, speed=4)
            if stable_gesture == "thumbs_up":
                state = "fighting"

        elif state == "fighting":
            # player action
            player_move = None
            if stable_gesture and now - player_action_time > ACTION_COOLDOWN:
                player_move        = stable_gesture
                player_action_time = now
                pm = {"fist": PUNCH_POSE, "peace": KICK_POSE, "palm": BLOCK_POSE}
                player.set_pose(pm.get(player_move, IDLE_POSE), speed=14)
                player._return_time = now + 0.45

            if now > player._return_time:
                player.set_pose(IDLE_POSE, speed=5)
                player._return_time = float("inf")

            # ai opp action
            ai_move = ai.update(difficulty)
            if ai_move:
                am = {"fist": PUNCH_POSE, "peace": KICK_POSE, "palm": BLOCK_POSE}
                opponent.set_pose(am.get(ai_move, IDLE_POSE), speed=12)
            else:
                opponent.set_pose(IDLE_POSE, speed=4)

            # player attacks opponent
            if player_move in ("fist", "peace", "palm"):
                if player_move == "palm":
                    if ai_move in ("fist", "peace"):
                        hit, dmg, msg = resolve_combat(ai_move, "palm")
                        if not hit:
                            action_labels.append(["BLOCKED!", player.cx, player.cy-150, 1.1, COLORS["block_flash"]])
                        else:
                            player.hp -= dmg
                            player.flash(COLORS["hit_flash"])
                            screen_shake = 0.5 if dmg >= KICK_DMG else 0.25
                            for _ in range(20): particles.append(HitParticle(player.cx, player.cy-80, (255,80,80)))
                    else:
                        opponent.knockback_vx = 5.0 * -1 * opponent.facing
                        opponent.hp -= PUSH_DMG
                        opponent.flash(COLORS["push_color"], 0.2)
                        action_labels.append(["PUSH!", opponent.cx, opponent.cy-140, 1.1, COLORS["push_color"]])
                else:
                    hit, dmg, msg = resolve_combat(player_move, ai_move)
                    if hit:
                        opponent.hp -= dmg
                        opponent.flash(COLORS["hit_flash"])
                        opponent.knockback_vx = 4.0 * -1 * opponent.facing
                        screen_shake = 0.55 if dmg >= KICK_DMG else 0.28
                        for _ in range(30): particles.append(HitParticle(opponent.cx, opponent.cy-80, COLORS["hit_flash"]))
                        lbl = "KICK!" if player_move == "peace" else "PUNCH!"
                        col = COLORS["kick_color"] if player_move == "peace" else COLORS["punch_color"]
                        action_labels.append([lbl, opponent.cx, opponent.cy-155, 1.2, col])
                        if msg: action_labels.append([msg, opponent.cx, opponent.cy-195, 1.0, (255,255,100)])
                    else:
                        if msg: action_labels.append([msg, player.cx, player.cy-150, 1.0, COLORS["block_flash"]])

            # ai opp attacks player
            if ai_move and ai_move != "palm":
                if ai._last_hit_move != ai.move_end_time:
                    ai._last_hit_move = ai.move_end_time
                    hit, dmg, msg = resolve_combat(ai_move, stable_gesture)
                    if hit:
                        player.hp -= dmg
                        player.flash(COLORS["hit_flash"])
                        player.knockback_vx = 4.0 * -1 * player.facing
                        screen_shake = 0.55 if dmg >= KICK_DMG else 0.28
                        for _ in range(25): particles.append(HitParticle(player.cx, player.cy-80, (255,80,80)))
                        lbl = "AI KICK!" if ai_move == "peace" else "AI PUNCH!"
                        action_labels.append([lbl, player.cx, player.cy-155, 1.2, (255,100,100)])
                    else:
                        if msg: action_labels.append([msg, player.cx, player.cy-150, 1.0, COLORS["block_flash"]])

            player.hp   = max(0, player.hp)
            opponent.hp = max(0, opponent.hp)

            if player.hp <= 0 or opponent.hp <= 0:
                winner_text = "YOU WIN!" if opponent.hp <= 0 else "OPPONENT WINS!"
                if opponent.hp <= 0: player_wins += 1
                else: ai_wins += 1
                state = "round_end"
                round_end_timer = 3.5

        elif state == "round_end":
            round_end_timer -= dt
            if round_end_timer <= 0:
                if player_wins >= 2 or ai_wins >= 2:
                    state = "game_over"
                else:
                    round_num  += 1
                    difficulty  = 1.0 + (round_num-1)*0.3
                    player.hp   = 100
                    opponent.hp = 100
                    player.cx   = float(W//4)
                    opponent.cx = float(W*3//4)
                    player.set_pose(IDLE_POSE)
                    opponent.set_pose(IDLE_POSE)
                    particles.clear()
                    action_labels.clear()
                    state = "waiting"

        elif state == "game_over":
            if stable_gesture == "thumbs_up":
                player   = Stickman(W//4,   H-80, COLORS["player"],   facing=1)
                opponent = Stickman(W*3//4, H-80, COLORS["opponent"], facing=-1)
                ai = AIOpponent()
                player_wins = ai_wins = 0
                round_num = 1; difficulty = 1.0
                particles.clear(); action_labels.clear()
                state = "waiting"

        # update objects 
        player.update(dt)
        opponent.update(dt)
        particles = [p for p in particles if p.life > 0]
        for p in particles: p.update()
        action_labels = [l for l in action_labels if l[3] > 0]
        for l in action_labels: l[3] -= dt; l[2] -= dt * 35
        if screen_shake > 0: screen_shake = max(0.0, screen_shake - dt * 3)

        # screen shake 
        if sdx or sdy:
            M = np.float32([[1,0,sdx],[0,1,sdy]])
            frame = cv2.warpAffine(frame, M, (W, H))

        # draw
        cv2.line(frame, (0, H-60), (W, H-60), (80,80,80), 2)
        player.draw(frame)
        opponent.draw(frame)
        for p in particles: p.draw(frame)
        for lbl in action_labels:
            draw_action_label(frame, lbl[0], lbl[1], lbl[2], lbl[4], min(1.0, lbl[3]))

        # HUD
        draw_health_bar(frame, 30, 20, 340, 30, player.hp, "YOU", flip=False)
        draw_health_bar(frame, W-370, 20, 340, 30, opponent.hp, "CPU", flip=True)
        rtxt = f"ROUND {round_num}   YOU {player_wins} - {ai_wins} CPU"
        draw_center(frame, rtxt, 75, scale=0.75, color=(220,220,220), thick=1)

        # gesture label bottom left
        glabels = {"fist": ("PUNCH", COLORS["punch_color"]), "peace": ("KICK", COLORS["kick_color"]), "palm": ("BLOCK", COLORS["push_color"])}
        if stable_gesture in glabels:
            gt, gc = glabels[stable_gesture]
            cv2.putText(frame, gt, (22, H-22), cv2.FONT_HERSHEY_DUPLEX, 1.2, COLORS["black"], 5, cv2.LINE_AA)
            cv2.putText(frame, gt, (22, H-22), cv2.FONT_HERSHEY_DUPLEX, 1.2, gc, 2, cv2.LINE_AA)

        draw_legend(frame)

        # overlays
        if state == "waiting":
            ov = frame.copy()
            cv2.rectangle(ov, (0,0), (W,H), (0,0,0), -1)
            frame = cv2.addWeighted(ov, 0.4, frame, 0.6, 0)
            draw_center(frame, f"ROUND {round_num}", H//2-60, scale=2.2, color=(255,220,50))
            draw_center(frame, "Show  THUMBS UP  to START", H//2+15, scale=0.95, color=(200,200,200), thick=2)

        elif state == "round_end":
            ov = frame.copy()
            cv2.rectangle(ov, (0,0),(W,H),(0,0,0),-1)
            frame = cv2.addWeighted(ov,0.45,frame,0.55,0)
            col = COLORS["player"] if "YOU WIN" in winner_text else COLORS["opponent"]
            draw_center(frame, winner_text, H//2, scale=2.2, color=col)
            draw_center(frame, "Next round starting...", H//2+70, scale=0.9, color=(200,200,200), thick=2)

        elif state == "game_over":
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(W,H),(0,0,0),-1)
            frame = cv2.addWeighted(ov,0.55,frame,0.45,0)
            if player_wins > ai_wins:
                draw_center(frame, "VICTORY!", H//2-50, scale=2.8, color=COLORS["player"])
            else:
                draw_center(frame, "DEFEATED!", H//2-50, scale=2.8, color=COLORS["opponent"])
            draw_center(frame, f"YOU {player_wins}  -  {ai_wins} CPU", H//2+30, scale=1.2, color=(220,220,220), thick=2)
            draw_center(frame, "Show THUMBS UP to play again", H//2+100, scale=0.9, color=(180,180,180), thick=2)

        cv2.imshow("Stickman Fight AR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()