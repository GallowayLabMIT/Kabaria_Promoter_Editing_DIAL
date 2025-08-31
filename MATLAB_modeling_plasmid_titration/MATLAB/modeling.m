%% Modeling for ZF37 titration
%%
filepath_noCre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_noCre_ZF37titration.xlsx';
filepath_Cre = '/Users/yunbeenbae/Documents/GitHub/Cre-loxP-ZF-promoters/flow_analysis_scripts/2024_promoter_editing_paper/datafiles/modeling_Cre_ZF37titration.xlsx';

data_noCre = readmatrix(filepath_noCre);
data_Cre = readmatrix(filepath_Cre);

%% for all data points
% reporter - 1
% ZF37 - 2
% well - 3
% mGL-A - 4
% mCherry-A - 5
% iRFP670-A - 6
% bioreplicate - 7
% log10 mGL-A - 8
% Normalized_mGL-A_gmean - 9
% Normalized_mCherry-A_gmean - 10

%% for only summary plot data
% reporter - 1
% ZF37 - 2
% bioreplicate - 3
% mGL-A_gmean - 4
% mCherry-A_gmean - 5
% Fraction - 6
% Count - 7
% Normalized_mGL-A_gmean - 8
% Normalized_mCherry-A_gmean - 9
%% 
x_zf_noCre = data_noCre(:,9); % mCherry-A
y_rna_noCre = data_noCre(:,8); % mGL-A

x_zf_Cre = data_Cre(:,9); % mCherry-A
y_rna_Cre = data_Cre(:,8); % mGL-A
%%
zf_burden = 7; % selected upon visual inspection of graph
positions = find(x_zf_Cre > zf_burden);

% remove the points related to burden
x_zf_Cre(positions) = [];
y_rna_Cre(positions) = [];

%%

objective_function = @(params, zf) f(params, zf, y_rna_noCre);

params_guess = [0.5, 1, 0.5, 0.5];

params_noCre = lsqcurvefit(objective_function, params_guess, x_zf_noCre, y_rna_noCre);

%%
predictions_noCre = f2(params_noCre, x_zf_noCre);


%%
objective_function = @(params, zf) f(params, zf, y_rna_Cre);

params_guess = [13, 1, 0.5, 0.5];
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
plot(x_zf_Cre, y_rna_Cre, 'go', 'DisplayName', 'Cre Data', 'MarkerSize',3);
hold on;
plot(x_zf_Cre, predictions_Cre, 'g-', 'DisplayName', 'Cre Model Predictions');
hold off;

%%
function residuals = f(params, zf, rna)

% unpack constants
k_cat = params(1);
n = params(2);
K_M = params(3);
beta = params(4);

predictions = k_cat / beta * zf.^n ./ (K_M + zf.^n);

residuals = predictions - rna;

end

function predictions = f2(params, zf)

% unpack constants
k_cat = params(1);
n = params(2);
K_M = params(3);
beta = params(4);

predictions = k_cat / beta * zf.^n ./ (K_M + zf.^n);

end