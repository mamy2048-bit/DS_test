# dsc_control: Nav2(MPPI) 명령을 MPC + ESO로 가공해 로봇에 보내는 속도 루프

Nav2가 낸 속도 명령을 그대로 드라이버에 보내지 않고, **로봇이 실제로 그 속도로 움직이도록** MPC + ESO로 가공한 뒤 `/cmd_vel`로 내보낸다.

**왜 필요한가.** Humble MPPI는 "명령한 속도가 다음 0.1 s에 그대로 나온다"고 가정한다(`motion_models.hpp`: `vx[t+1] = cvx[t]`). 실제 로봇은 지연이 있고 저속에서 오차가 생긴다. MPC + ESO는 MPPI가 원한 속도가 실제로 나오게 만드는 안쪽 속도 루프다.

| `mode` | 동작 |
|---|---|
| `passthrough` | MPPI 명령을 한계값으로만 자른다 (기준선) |
| `nominal` | MPC: 로봇 모델로 "기준 속도가 실제로 나오는" 명령을 계산 |
| `offset_free` | MPC + ESO: 모델과 실제의 차이(마찰, 경사, 모델 오차)를 외란으로 추정해 보정 |

세 mode 모두 같은 토픽 경로를 지나므로 지연 조건이 같다.

---

## 1. 구조

```
MPPI ─/cmd_vel_nav→ velocity_smoother ─/cmd_vel_smoothed→ collision_monitor ─/cmd_vel_ref→ dsc_mpc_node ─/cmd_vel→ m2 드라이버
                                                                                   ▲
                                                                          /odom (실제 속도, 10 Hz)
```
- `dsc_mpc_node` 하나에 MPC와 ESO가 함께 들어 있다. `mode`로 사용 여부를 고른다.
- collision_monitor 출력만 `/cmd_vel_ref`로 바꾼다. launch 실행 시점에만 바꾸고, DS_navigation의 `params.yaml` 원본은 수정하지 않는다.
- 제어 주기: `/odom` 1회 = 제어 1스텝(0.1 s). 별도 타이머가 없으므로 `/odom`이 10 Hz여야 한다.
- **기준값이 정확히 0이면 MPC를 거치지 않고 즉시 0을 보낸다**(`stop_on_zero_ref`). 목표 도착, 충돌 정지, Nav2 종료가 여기에 해당한다.
- 기준값이나 `/odom`이 0.3 s(`timeout_s`) 넘게 끊기면 0을 5번 보내고 조용해진다.
- 노드가 꺼질 때(Ctrl-C, kill)는 0을 5번 보내고 종료한다. **m2 드라이버에는 cmd_vel 타임아웃이 없어** 마지막 명령을 계속 실행하므로, `kill -9`나 전원 차단 시에는 로봇이 멈추지 않을 수 있다.

## 2. 코드 구성

위에서 아래 순서로 읽으면 된다. 각 파일 맨 위에 `[인수인계]` 요약 주석이 있다.

| 파일 (`dsc_control/`) | 역할 |
|---|---|
| `launch/nav2_mpc.launch.py` | **진입점.** DS_navigation의 Nav2를 띄우면서 collision_monitor 출력을 `/cmd_vel_ref`로 돌리고, MPC 노드를 함께 띄운다 |
| `launch/mpc.launch.py` | MPC 노드만 띄운다 (`sim:=true`면 책상 시뮬레이션) |
| `dsc_control/nodes/mpc_node.py` | ROS 입출력. 언제 계산하고 언제 멈추는지, `/dsc/debug` 로그 |
| `dsc_control/core/controller.py` | 제어 1스텝의 순서: 상태 → ESO → 목표입력 → MPC → 한계 → ESO 전진. mode 3가지 분기 |
| `dsc_control/core/model.py` | 로봇 응답 모델 `y[k+1] = a·y + b·u[k-delay]`와 상태공간 변환 |
| `dsc_control/core/observer.py` | ESO: 축별 일정 외란 d 추정 |
| `dsc_control/core/target.py` | 정상상태 목표입력 ū, 최종 명령 한계(`limit_command`) |
| `dsc_control/core/mpc.py` | condensed QP 구성과 OSQP 풀이 |
| `dsc_control/core/config.py` | yaml → 설정 객체 |
| `dsc_control/nodes/stop.py` | 종료 시 0 명령을 확실히 내보내는 도우미 |
| `dsc_control/nodes/sim_robot.py`, `core/sim.py` | 로봇 없이 시험하는 1차 모델 플랜트 |
| `config/nav.yaml` | **Nav2 주행용 설정** (4절) |
| `config/default.yaml` | 책상 시뮬레이션용 초기 모델. Nav2 아래에서는 쓰지 않는다 |
| `test/` | 단위 테스트 (ROS 없이 `pytest`로 실행) |

`analysis/` (ROS 패키지 아님, 빌드되지 않음): `identify.py` 모델 식별(5절), `metrics.py` 속도 추종 오차, `bag_io.py` bag 읽기.

코드에 나오는 `require_gate`, `/dsc/gate_state`, `/cmd_vel_dock`, `dsc_docking`은 이 저장소에 없는 별도 도킹 패키지용이다. Nav2 주행에서는 쓰지 않는다(`require_gate=false`).

---

## 3. 설치 (Ubuntu 22.04 + ROS2 Humble)

이 저장소에는 MPC/ESO 부분만 들어 있다. 위치추정·Nav2와 로봇 드라이버는 아래 두 저장소를 쓴다(GitHub 접근 권한 필요).

| 경로 | 내용 |
|---|---|
| `~/ds_navigation/DS_navigation` | `ISR-Robotics-Lab/DS_navigation`. isr_navigation(Nav2 설정, 지도) + lidar_localization |
| `~/isr_tut` | `ISR-Robotics-Lab/isr_tut`. 로봇 드라이버(isr_robot_driver)·Velodyne·Xsens. **여기의 isr_navigation 은 구버전**이므로 source 순서를 지킨다 |
| `~/dsc_test_ws/src/DS_test` | 이 저장소 |

```bash
# 0) 의존 저장소 (각 저장소 README대로 빌드)
git clone git@github.com:ISR-Robotics-Lab/DS_navigation.git ~/ds_navigation/DS_navigation
git clone git@github.com:ISR-Robotics-Lab/isr_tut.git ~/isr_tut

# 1) 이 저장소
mkdir -p ~/dsc_test_ws/src && cd ~/dsc_test_ws/src
git clone git@github.com:mamy2048-bit/DS_test.git

# 2) Python 의존성 (numpy, scipy, yaml은 ROS와 함께 설치됨)
pip install --user osqp==0.6.7              # MPC의 QP solver (필수, 1.x는 API가 다름)
pip install --user rosbags pandas           # analysis/ 용 (선택)

# 3) 빌드
source /opt/ros/humble/setup.bash && source ~/ds_navigation/DS_navigation/install/setup.bash
cd ~/dsc_test_ws && colcon build --symlink-install
```

**매 터미널** 이 순서로 source 한다. `~/isr_tut`의 구버전 isr_navigation보다 DS_navigation이 우선하도록 **DS_navigation을 나중에** source 한다. zsh면 `setup.zsh`를 쓴다.
```bash
source /opt/ros/humble/setup.bash
source ~/isr_tut/install/setup.bash                        # 드라이버·Velodyne·Xsens
source ~/ds_navigation/DS_navigation/install/setup.bash     # isr_navigation·lidar_localization (우선)
source ~/dsc_test_ws/install/setup.bash
```

**설치 확인**
```bash
cd ~/dsc_test_ws/src/DS_test && python3 -m pytest -q     # 39 passed
ros2 pkg prefix isr_navigation                            # ds_navigation/DS_navigation 경로여야 한다
```

로봇 없이 확인: `ros2 launch dsc_control mpc.launch.py mode:=offset_free sim:=true` 후 `/cmd_vel_ref`에 Twist를 발행하면 `/dsc/debug`, `/odom`이 나온다.

---

## 4. 실행

터미널마다 3절의 source 네 줄을 먼저 실행한다.
```bash
# 1) 로봇 드라이버 + 센서 (publish_tf=False)
ros2 launch isr_robot_driver m2_manual_launch.py
#    자율주행 전에 키보드 `a` 로 autonomous mode 로 바꾼다
#    (manual mode 는 /cmd_vel 을 50 Hz 로 계속 쏘므로 MPC 와 충돌한다)

# 2) 위치추정 (DS_navigation 모드 C) — RViz 의 "2D Pose Estimate" 로 초기 위치를 준다
ros2 launch lidar_localization run.launch.py \
  map_file:=$HOME/ds_navigation/DS_navigation/src/lidar_localization/map_3D/GlobalMap.pcd

# 3) Nav2 + MPC 노드 (Nav2를 따로 켜지 않는다)
ros2 launch dsc_control nav2_mpc.launch.py mode:=offset_free
#    지도 기본값은 isr_navigation 의 map_2D/map.yaml. 다른 지도면 map:=<...>/map.yaml

# 4) goal 은 RViz 또는 isr_navigation 의 goal_publisher_gui
```

**모드 변경.** 로봇이 멈춰 있을 때만 된다(주행 중에는 거부).
```bash
ros2 param set /dsc_mpc_node mode offset_free   # passthrough | nominal | offset_free
```

**실행 직후 확인**
- [ ] `ros2 topic info /cmd_vel`의 발행자가 **1개(`dsc_mpc_node`)**인가
- [ ] `ros2 topic hz /odom`이 약 10 Hz인가
- [ ] `ros2 param get /collision_monitor cmd_vel_out_topic`이 `/cmd_vel_ref`인가
- [ ] 노드 로그에 `config=.../nav.yaml mode=... stop_on_zero_ref=True`가 찍혔는가

**금지**: 원래 isr_navigation의 Nav2 launch나 다른 컨트롤러와 함께 켜지 않는다. `/cmd_vel` 발행자가 둘이 된다.

**`/dsc/debug` 배열** (Float64MultiArray, 제어 스텝마다):
`[0] t_odom [1:3] ref [3:5] meas [5:7] u [7:9] u_bar [9:11] y_pred_1 [11] solve_ms [12] status_ok [13] mode_id [14:16] d_hat`

---

## 5. 파라미터

### 5.1 `nav2_mpc.launch.py` 인자
| 인자 | 기본값 | 의미 |
|---|---|---|
| `mode` | passthrough | passthrough / nominal / offset_free |
| `vx_max` | 0.35 | 전진 상한 [m/s]. MPPI `FollowPath.vx_max`와 MPC `v_max`를 **같이** 바꾼다 |
| `xy_goal_tolerance` | 0.08 | Nav2 도착 판정 반경 [m] |
| `yaw_goal_tolerance` | 0.10 | Nav2 도착 판정 헤딩 [rad] |
| `goal_angle_threshold` | 0.2 | MPPI GoalAngleCritic이 goal 헤딩을 맞추기 시작하는 거리 [m] (params.yaml 0.5) |
| `mpc_config` | config/nav.yaml | MPC 설정 파일 |
| `params_file`, `map` | isr_navigation 기본값 | Nav2 설정, 2D 지도 |

### 5.2 `config/nav.yaml`
| 항목 | 의미 |
|---|---|
| `model.v`, `model.w` | 축별 응답 모델 `a, b, delay`. 시정수 = −dt/ln a, 정상 이득 = b/(1−a). **로봇·드라이버·바닥·하중이 바뀌면 다시 식별한다 (5.3)** |
| `limits` | MPPI·velocity_smoother 한계와 같게 맞춘다. `v_min`은 MPPI `vx_min`과 같아야 한다(collision_monitor가 MPC 앞단에 있어 MPC 출력은 충돌 검사를 받지 않는다) |
| `mpc.q / r / s` | 추종 오차 / 정상 입력에서 벗어남 / 명령 변화량 가중치. q와 s는 비율만 의미가 있다 |
| `mpc.horizon` | 예측 스텝 수 (20 = 2 s) |
| `observer.bandwidth` | ESO 외란 추정 속도 [rad/s]. 크면 빠르지만 진동·잡음도 외란으로 본다 |
| `observer.max_correction` | 제어에 쓰는 ESO 보정 상한 (명령 단위 v m/s, w rad/s). 추정값 자체는 그대로 기록된다 |

### 5.3 모델 재식별과 조정 순서
1. `mode:=passthrough`로 평소처럼 주행하며 `/cmd_vel`, `/odom`을 bag으로 기록한다. 직선(최고 속도 도달), 회전, 후진이 섞이게 한다.
2. 식별:
   ```bash
   cd ~/dsc_test_ws/src/DS_test
   python3 -m analysis.identify <bag 경로> --cmd /cmd_vel --odom /odom --max-delay 8
   ```
   출력된 `v`, `w` 줄을 `nav.yaml`의 `model`에 넣는다.
3. `nominal`로 주행하며 `/dsc/debug`의 `u`(명령)와 `meas`(실제)를 비교한다.

   | 증상 | 조정 |
   |---|---|
   | 명령이 튀거나 출렁임 | `s` ↑ 또는 `q` ↓ |
   | 실제 속도가 기준을 늦게 따라감 | `q` ↑ 또는 `s` ↓ |
   | 회전이 불안정 | `q_w` ↓, `s_w` ↑ |
   | `solve_ms`가 50 ms에 가까움 | `horizon` ↓ |
4. `offset_free`로 주행하며 `d_hat`과 추종 오차를 본다.

   | 증상 | 조정 |
   |---|---|
   | 일정한 속도 오차가 오래 남음 | `bandwidth` ↑ (조금씩) |
   | `d_hat`과 명령이 잘게 떨림 | `bandwidth` ↓ |
   | goal 전환·정지 직후 반대 방향 명령 | `max_correction` ↓, 모델 재확인 |

### 5.4 노드 파라미터 (`dsc_mpc_node`)
| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `mode` | passthrough | 실행 중 변경 가능(정지 상태에서만) |
| `config_path` | config/default.yaml | 설정 파일. nav2_mpc.launch.py는 nav.yaml을 넘긴다 |
| `v_max` | 0.0 | > 0이면 설정 파일의 `limits.v_max`를 덮어쓴다 |
| `stop_on_zero_ref` | true | 기준값 0이면 즉시 0 + 제어기 초기화 |
| `timeout_s` | 0.3 | 기준값·`/odom`이 이 시간 끊기면 정지 |
| `idle_zero_count` | 5 | 멈출 때 0을 보내는 횟수 |
| `cmd_vel_topic` | /cmd_vel | 출력 토픽 |
| `require_gate` | false | 도킹 패키지 전용 |

---

## 6. 알아둘 점
- **순항 속도에서는 ESO 보정이 안 걸린다.** 명령 상한(`v_max`)이 MPPI 상한과 같으므로, 기준값이 상한일 때는 ESO가 그 위로 보정할 수 없다. 효과는 감속·접근 구간에서 나온다.
- **모델이 실제와 많이 다르면 MPC + ESO가 진동을 만든다.** 특히 `delay`를 빠뜨리면 그렇다. 이상하면 먼저 `passthrough`와 비교하고 모델을 다시 식별한다.
- Nav2가 정지할 때 `/cmd_vel_ref`가 아주 작은 값(정확히 0이 아님)이면 즉시 정지 대신 MPC 감속으로 처리된다.
- QP가 실패하면 `max_failures`(3)회까지 직전 계획을 이어 쓰고, 그 뒤에는 0을 보낸다.
