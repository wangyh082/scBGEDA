clear all
clc
rand('state',0)
alpha=1e-5
hvgs=2000
MeanNMI= zeros(11,15);
MeanARI = zeros(11,15);
MedianNMI= zeros(11,15);
MedianARI = zeros(11,15);
for num=1:1:15
    num
    switch num
        case 1
            dataName = 'Adam';
%         case 2
%            dataName = 'Klein';    
%         case 3
%             dataName = 'Muraro';
%         case 4
%            dataName = 'Bach';
%         case 5
%             dataName = 'Quake_10x_Bladder';
%         case 6
%             dataName = 'Quake_10x_Limb_Muscle';
%         case 7
%             dataName = 'Quake_10x_Trachea';
%         case 8
%             dataName = 'Quake_Smart-seq2_Diaphragm';
%         case 9
%             dataName = 'Quake_Smart-seq2_Heart';
%         case 10
%             dataName = 'Quake_Smart-seq2_Limb_Muscle';
%         case 11
%             dataName = 'Quake_Smart-seq2_Lung';
%         case 12
%             dataName = 'Quake_Smart-seq2_Trachea';
%         case 13
%             dataName = 'Tosches_turtle';          
%         case 14
%             dataName = 'Wang_Lung';
%         case 15
%             dataName = 'Young';
    end
random_seed=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10000];


for seed_index=1:length(random_seed)
gt = [];
fea = [];

load(['/home/mdata/',dataName,'_seed_',num2str(random_seed(seed_index)),'_',num2str(hvgs),'_',num2str(alpha),'.mat'])

label=double(label);
fea=data;

gt=label+1;

[N, d] = size(fea);
%% Set up
m = 100; % Ensemble size
k = numel(unique(gt)); % The number of clusters
cntTimes = 3;

nmiScores = zeros(cntTimes,1);
ariScores = zeros(cntTimes,1);
disp('.');
disp(['N = ',num2str(N)]);
disp('.');
for runIdx = 1:cntTimes
    disp('**************************************************************');
    disp(['Run ', num2str(runIdx),':']);
    disp('**************************************************************');
    

    % Or you can set up parameters by yourself.
     bcsLowK = 2;
     bcsUpK = 60;
     Label = BGEC(fea, k, m, bcsLowK, bcsUpK);
    
    disp('--------------------------------------------------------------'); 
    nmiScores(runIdx) = computeNMI(Label,gt);
    ariScores(runIdx) = rand_index(Label,gt, 'adjusted');
    
    disp(['The NMI score at Run ',num2str(runIdx), ': ',num2str(nmiScores(runIdx))]);   
    disp(['The ARI score at Run ',num2str(runIdx), ': ',num2str(ariScores(runIdx))]);   
    disp('--------------------------------------------------------------');
end

MeanNMI(seed_index,num)=max(nmiScores);
MeanARI(seed_index,num)=max(ariScores);
MedianNMI(seed_index,num)=max(nmiScores);
MedianARI(seed_index,num)=max(ariScores);
end
MeanNMI(length(random_seed)+1,num)=mean(MeanNMI(1:length(random_seed),num));
MeanARI(length(random_seed)+1,num)=mean(MeanARI(1:length(random_seed),num));
MedianNMI(length(random_seed)+1,num)=median(MedianNMI(1:length(random_seed),num));
MedianARI(length(random_seed)+1,num)=median(MedianARI(1:length(random_seed),num));

end
MeanNMI = array2table(MeanNMI);
MeanNMI.Properties.VariableNames(1:15) = {'Adam','Klein','Muraro','Bach','Quake_10x_Bladder','Quake_10x_Limb_Muscle','Quake_10x_Trachea',...
    'Quake_Smart-seq2_Diaphragm','Quake_Smart-seq2_Heart','Quake_Smart-seq2_Limb_Muscle','Quake_Smart-seq2_Lung','Quake_Smart-seq2_Trachea','Tosches_turtle','Wang_Lung','Young'}
MeanARI = array2table(MeanARI);
MeanARI.Properties.VariableNames(1:15) = {'Adam','Klein','Muraro','Bach','Quake_10x_Bladder','Quake_10x_Limb_Muscle','Quake_10x_Trachea',...
    'Quake_Smart-seq2_Diaphragm','Quake_Smart-seq2_Heart','Quake_Smart-seq2_Limb_Muscle','Quake_Smart-seq2_Lung','Quake_Smart-seq2_Trachea','Tosches_turtle','Wang_Lung','Young'}
MedianNMI = array2table(MedianNMI);
MedianNMI.Properties.VariableNames(1:15) = {'Adam','Klein','Muraro','Bach','Quake_10x_Bladder','Quake_10x_Limb_Muscle','Quake_10x_Trachea',...
    'Quake_Smart-seq2_Diaphragm','Quake_Smart-seq2_Heart','Quake_Smart-seq2_Limb_Muscle','Quake_Smart-seq2_Lung','Quake_Smart-seq2_Trachea','Tosches_turtle','Wang_Lung','Young'}
MedianARI = array2table(MedianARI);
MedianARI.Properties.VariableNames(1:15) = {'Adam','Klein','Muraro','Bach','Quake_10x_Bladder','Quake_10x_Limb_Muscle','Quake_10x_Trachea',...
    'Quake_Smart-seq2_Diaphragm','Quake_Smart-seq2_Heart','Quake_Smart-seq2_Limb_Muscle','Quake_Smart-seq2_Lung','Quake_Smart-seq2_Trachea','Tosches_turtle','Wang_Lung','Young'}
filename=['KNN_',num2str(KNN),'NC+1','+hvgs',num2str(hvgs),'+alpha',num2str(alpha),'.xlsx']
writetable(MeanNMI,filename,'FileType','spreadsheet','Sheet','MeanNMI')
writetable(MeanARI,filename,'FileType','spreadsheet','Sheet','MeanARI')
writetable(MedianNMI,filename,'FileType','spreadsheet','Sheet','MedianNMI')
writetable(MedianARI,filename,'FileType','spreadsheet','Sheet','MedianARI')