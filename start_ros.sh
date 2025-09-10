#!/bin/bash

# -------- Preparation --------
# Function to get current pts (pseudo terminals)
get_current_pts_set() {
    ps -ef | grep "bash -c" | grep -v grep | awk '{print $6}' | sort
}

# Initialize tracking variables
BEFORE_PTS=$(get_current_pts_set)
PTS_LIST=()

# Function to detect newly spawned terminals
get_new_pts() {
    sleep 1  # Give time for terminal to initialize
    local AFTER_PTS=$(get_current_pts_set)
    local NEW_PTS=$(comm -13 <(echo "$BEFORE_PTS") <(echo "$AFTER_PTS"))
    BEFORE_PTS="$AFTER_PTS"
    for pts in $NEW_PTS; do
        PTS_LIST+=("$pts")
    done
}

# -------- Cleanup Function (for q or Ctrl+C) --------
cleanup() {
    echo -e "\n🚨 Shutting down all spawned terminals..."

    for pts in "${PTS_LIST[@]}"; do
        echo "Killing terminal $pts..."
        pkill -t "$pts"
    done

    # Kill tmux session if exists
    # if tmux has-session -t sfm-arena 2>/dev/null; then
    #     tmux kill-session -t sfm-arena
    #     echo "✅ tmux session 'sfm-arena' killed."
    # else
    #     echo "⚠️ No tmux session found."
    # fi

    tmux kill-server
    echo "✅ All tmux sessions killed."

    echo "✅ All cleaned up. Bye!"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

# -------- Launch terminal tasks --------
gnome-terminal -- bash -c "roscore; exec bash" & get_new_pts
gnome-terminal -- bash -c "roslaunch --wait arena_bringup start_arena.launch simulator:=gazebo model:=jackal map_file:=small_warehouse tm_obstacles:=scenario tm_robots:=scenario scenario_file:=default.json entity_manager:=pedsim; exec bash" & get_new_pts
# gnome-terminal -- bash -c "roslaunch --wait tvss_nav start_arena_sfm.launch simulator:=gazebo model:=jackal map_file:=arena_hospital_small tm_obstacles:=scenario tm_robots:=scenario scenario_file:=v10.json entity_manager:=pedsim; exec bash" & get_new_pts
gnome-terminal -- bash -c "roslaunch --wait rosbridge_server rosbridge_websocket.launch; exec bash" & get_new_pts
gnome-terminal -- bash -c "roslaunch --wait teleop_twist_joy teleop.launch joy_config:=xbox joy_dev:=/dev/input/js0; exec bash" & get_new_pts   
gnome-terminal -- bash -c "rosrun topic_tools relay /cmd_vel /jackal/cmd_vel; exec bash" & get_new_pts
sleep 10
gnome-terminal -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate vlm_nav && python src/vlm_nav_node.py; exec bash" & get_new_pts

# -------- Exit handling --------

echo "Press 'q' to quit, close all spawned terminals, and kill tmux session if running (or press Ctrl+C)..."

while true; do
    read -n 1 key
    if [[ $key == "q" ]]; then
        cleanup  
    fi
done
