% Improved gbike policy iteration (aligned with first implementation, preserves colormaps)
clear all
close all

max_bikes = 20;
max_move = 5;
gamma = 0.9;

rent1 = 3; ret1 = 3;
rent2 = 4; ret2 = 2;

r_reward = 10;
move_cost_per = 2;
parking_penalty = 4;

policy = zeros(max_bikes+1, max_bikes+1);
V = zeros(max_bikes+1, max_bikes+1);

max_poisson = 20;
n = 0:max_poisson;
P_rent1 = exp(-rent1) .* (rent1 .^ n) ./ factorial(n);
P_rent2 = exp(-rent2) .* (rent2 .^ n) ./ factorial(n);
P_ret1  = exp(-ret1)  .* (ret1  .^ n) ./ factorial(n);
P_ret2  = exp(-ret2)  .* (ret2  .^ n) ./ factorial(n);

trunc_idx = find(cumsum(P_rent1) > 0.9999,1);
if isempty(trunc_idx), trunc_idx = max_poisson; end
trunc = max(trunc_idx,10);
trunc = min(trunc, max_poisson);

P_rent1 = P_rent1(1:trunc+1);
P_rent2 = P_rent2(1:trunc+1);
P_ret1  = P_ret1(1:trunc+1);
P_ret2  = P_ret2(1:trunc+1);

policy_stable = false;
iter = 0;
while ~policy_stable
    V = policy_evaluation(V, policy, max_bikes, max_move, P_rent1, P_rent2, P_ret1, P_ret2, r_reward, move_cost_per, parking_penalty, gamma);
    [policy, policy_stable] = policy_improvement(V, policy, max_bikes, max_move, P_rent1, P_rent2, P_ret1, P_ret2, r_reward, move_cost_per, parking_penalty, gamma);
    iter = iter + 1;
end

figure;
heatmap(policy, 'Colormap', parula);
title('Optimal Policy Heatmap');
xlabel('Bikes at Location 2');
ylabel('Bikes at Location 1');

figure;
heatmap(V, 'Colormap', hot);
title('Value Function Heatmap');
xlabel('Bikes at Location 2');
ylabel('Bikes at Location 1');


function [policy, stable] = policy_improvement(V, policy, max_bikes, max_move, P_r1, P_r2, P_t1, P_t2, r_reward, move_cost_per, parking_penalty, gamma)
stable = true;
old_policy = policy;
for i = 0:max_bikes
    for j = 0:max_bikes
        s1 = i; s2 = j;
        amin = -min([s2, max_move]);
        amax = min([s1, max_move]);
        best_val = -inf;
        best_a = policy(i+1,j+1);
        for a = amin:amax
            cost_move = compute_move_cost(a, move_cost_per);
            new_s1 = s1 - a;
            new_s2 = s2 + a;
            if new_s1 > 10, cost_move = cost_move + parking_penalty; end
            if new_s2 > 10, cost_move = cost_move + parking_penalty; end
            expected_reward = 0;
            expected_V = 0;
            for rent1 = 0:length(P_r1)-1
                p_r1 = P_r1(rent1+1);
                rentals1 = min(new_s1, rent1);
                prob1 = p_r1;
                for rent2 = 0:length(P_r2)-1
                    p_r2 = P_r2(rent2+1);
                    rentals2 = min(new_s2, rent2);
                    prob_r = prob1 * p_r2;
                    remaining1 = new_s1 - rentals1;
                    remaining2 = new_s2 - rentals2;
                    for ret1 = 0:length(P_t1)-1
                        p_t1 = P_t1(ret1+1);
                        for ret2 = 0:length(P_t2)-1
                            p_t2 = P_t2(ret2+1);
                            prob = prob_r * p_t1 * p_t2;
                            next1 = min(remaining1 + ret1, max_bikes);
                            next2 = min(remaining2 + ret2, max_bikes);
                            expected_reward = expected_reward + prob * ( (rentals1 + rentals2) * r_reward );
                            expected_V = expected_V + prob * V(next1+1, next2+1);
                        end
                    end
                end
            end
            val = expected_reward - cost_move + gamma * expected_V;
            if val > best_val
                best_val = val;
                best_a = a;
            end
        end
        policy(i+1,j+1) = best_a;
    end
end
if any(old_policy(:) ~= policy(:)), stable = false; end
end

function V = policy_evaluation(V, policy, max_bikes, max_move, P_r1, P_r2, P_t1, P_t2, r_reward, move_cost_per, parking_penalty, gamma)
theta = 1e-3;
while true
    delta = 0;
    V_old = V;
    for i = 0:max_bikes
        for j = 0:max_bikes
            a = policy(i+1,j+1);
            cost_move = compute_move_cost(a, move_cost_per);
            new_s1 = i - a;
            new_s2 = j + a;
            if new_s1 > 10, cost_move = cost_move + parking_penalty; end
            if new_s2 > 10, cost_move = cost_move + parking_penalty; end
            expected_reward = 0;
            expected_V = 0;
            for rent1 = 0:length(P_r1)-1
                p_r1 = P_r1(rent1+1);
                rentals1 = min(new_s1, rent1);
                for rent2 = 0:length(P_r2)-1
                    p_r2 = P_r2(rent2+1);
                    rentals2 = min(new_s2, rent2);
                    prob_r = p_r1 * p_r2;
                    remaining1 = new_s1 - rentals1;
                    remaining2 = new_s2 - rentals2;
                    for ret1 = 0:length(P_t1)-1
                        p_t1 = P_t1(ret1+1);
                        for ret2 = 0:length(P_t2)-1
                            p_t2 = P_t2(ret2+1);
                            prob = prob_r * p_t1 * p_t2;
                            next1 = min(remaining1 + ret1, max_bikes);
                            next2 = min(remaining2 + ret2, max_bikes);
                            expected_reward = expected_reward + prob * ((rentals1 + rentals2) * r_reward);
                            expected_V = expected_V + prob * V_old(next1+1, next2+1);
                        end
                    end
                end
            end
            new_v = expected_reward - cost_move + gamma * expected_V;
            delta = max(delta, abs(new_v - V_old(i+1,j+1)));
            V(i+1,j+1) = new_v;
        end
    end
    if delta < theta
        break;
    end
end
end

function cost = compute_move_cost(a, move_cost_per)
if a > 0
    cost = move_cost_per * max(0, a-1);
elseif a < 0
    cost = move_cost_per * abs(a);
else
    cost = 0;
end
end
