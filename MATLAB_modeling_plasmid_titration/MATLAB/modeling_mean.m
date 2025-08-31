%% Modeling for ZF37 titration
%%
filepath_noCre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_MEAN_noCre_ZF37titration.xlsx';
filepath_Cre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_MEAN_Cre_ZF37titration.xlsx';

data_noCre = readmatrix(filepath_noCre);
data_Cre = readmatrix(filepath_Cre);

%% for mean of normalized gmean data

% zf_conc (target) - 1
% mGL - 2
% mCherry - 3

%% 
x_zf_noCre = data_noCre(:,3); % mCherry-A
y_rna_noCre = data_noCre(:,2); % mGL-A

x_zf_Cre = data_Cre(:,3); % mCherry-A
y_rna_Cre = data_Cre(:,2); % mGL-A
%%
zf_burden = 7; % selected upon visual inspection of graph
positions = find(x_zf_Cre > zf_burden);

% remove the points related to burden
x_zf_Cre(positions) = [];
y_rna_Cre(positions) = [];

%%

objective_function = @(params, zf) f2(params, zf);

% k_cat_prime, K_M, alpha_prime
params_guess = [0.5, 0.5, 0.5];

params_noCre = lsqcurvefit(objective_function, params_guess, x_zf_noCre, y_rna_noCre);

%%
predictions_noCre = f2(params_noCre, x_zf_noCre);


%%
objective_function = @(params, zf) f2(params, zf);

params_guess = [0.5, 0.5, 0.5];
params_Cre = lsqcurvefit(objective_function, params_guess, x_zf_Cre, y_rna_Cre);

%%
predictions_Cre = f2(params_Cre, x_zf_Cre);


%%

% Plot the data and model predictions
plot(x_zf_noCre, y_rna_noCre, 'ro', 'DisplayName', 'No Cre Data', 'MarkerSize',3);
hold on;
plot(x_zf_noCre, predictions_noCre, 'r-', 'DisplayName', 'No Cre Model Predictions');
xlabel('mCherry-A Normalized by gmean of ZF37 0.125x');
ylabel('mGL-A  Normalized by gmean of ZF37 0.125x');
title('Scatterplot Normalized by ZF37 0.125x with Fitted model');
legend;
set(gca, 'XScale', 'linear'); % Set x-axis to logarithmic scale
set(gca, 'YScale', 'linear'); % Set y-axis to logarithmic scale
grid on;
hold on;
% Plot the data and model predictions
plot(x_zf_Cre, y_rna_Cre, 'bo', 'DisplayName', 'Cre Data', 'MarkerSize',3);
hold on;
plot(x_zf_Cre, predictions_Cre, 'g-', 'DisplayName', 'Cre Model Predictions');
hold off;

%%

labels = {'k_cat_prime','K_M','alpha_prime'};

disp('No Cre parameters')
disp('Index   Value   Label');
for i = 1:length(params_noCre)
    fprintf('%d\t\t%d\t\t%s\n', i, params_noCre(i), labels{i});
end

disp('Cre parameters')
disp('Index   Value   Label');
for i = 1:length(params_Cre)
    fprintf('%d\t\t%d\t\t%s\n', i, params_Cre(i), labels{i});
end

%%

function predictions = f2(params, zf)

% unpack constants
k_cat_prime = params(1);
K_M = params(2);
alpha_prime = params(3);

n=2;

predictions = k_cat_prime * zf.^n ./ (K_M + zf.^n) + alpha_prime;

end