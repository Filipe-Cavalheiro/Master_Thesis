centralized_agents = load("centralized_agent_pos.mat");
centralized_agents = centralized_agents.agents;

decentralized_agents = load("decentralized_agent_pos.mat");
decentralized_agents = decentralized_agents.agents;

figure
hold on;
scatter(centralized_agents(:,1), centralized_agents(:,2), 'x', 'g');

scatter(decentralized_agents(:,1), decentralized_agents(:,2), 'd', 'r');