%% Modeling for ZF37 titration

%% all summary plot, ZF37

% for only summary plot data
% reporter - 1
% ZF37 - 2
% bioreplicate - 3
% mGL-A_gmean - 4
% mCherry-A_gmean - 5
% Fraction - 6
% Count - 7
% Normalized_mGL-A_gmean - 8
% Normalized_mCherry-A_gmean - 9

% code starts here

%filepath_noCre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_noCre_ZF37titration.xlsx';
%filepath_Cre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_Cre_ZF37titration.xlsx';

%x_zf_noCre = data_noCre(:,9); % mCherry-A
%y_rna_noCre = data_noCre(:,8); % mGL-A

%x_zf_Cre = data_Cre(:,9); % mCherry-A
%y_rna_Cre = data_Cre(:,8); % mGL-A


%% mean of gmeans, ZF37

% for mean of normalized gmean data
% zf_conc (target) - 1
% mGL - 2
% mCherry - 3

% code starts here

filepath_noCre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_MEAN_noCre_ZF37titration.xlsx';
filepath_Cre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_MEAN_Cre_ZF37titration.xlsx';

data_noCre = readmatrix(filepath_noCre);
data_Cre = readmatrix(filepath_Cre);

x_zf_noCre = data_noCre(:,3); % mCherry-A
y_rna_noCre = data_noCre(:,2); % mGL-A

x_zf_Cre = data_Cre(:,3); % mCherry-A
y_rna_Cre = data_Cre(:,2); % mGL-A

%% removing burden points

% gate visually
zf_burden = 7; % selected upon visual inspection of graph
positions = find(x_zf_Cre > zf_burden);

% remove the points related to burden
x_zf_Cre(positions) = [];
y_rna_Cre(positions) = [];

%% initializing mCherry expression

x_zf_Cre = x_zf_Cre-x_zf_Cre(1);
x_zf_noCre = x_zf_noCre-x_zf_noCre(1);

%% fit the curve to noCre points

objective_function = @(params, zf) f2(params, zf);

% k_cat_prime, K_M, alpha_prime
params_guess = [1, 0.05, 0];

params_noCre = lsqcurvefit(objective_function, params_guess, x_zf_noCre, y_rna_noCre);

% calculate estimated reporter expression based on fitted parameters
predictions_noCre = f2(params_noCre, x_zf_noCre);

% repeat for Cre points
objective_function = @(params, zf) f2(params, zf);

% k_cat_prime, K_M, alpha_prime
params_guess = [14, 0.2, 0];
params_Cre = lsqcurvefit(objective_function, params_guess, x_zf_Cre, y_rna_Cre);

predictions_Cre = f2(params_Cre, x_zf_Cre);

%% Plot the data and model predictions

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

%% Display parameters

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

%% Functions

function predictions = f2(params, zf)

% unpack constants
k_cat_prime = params(1);
K_M = params(2);
alpha_prime = params(3);

% change to fix n
n=2;

predictions = k_cat_prime * zf.^n ./ (K_M + zf.^n) + alpha_prime;

end