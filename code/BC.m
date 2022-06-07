function []=BC(dataName, m, bcsLowK, bcsUpK, random_seed)
if nargin < 5
    random_seed=[1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10000];
end
if nargin < 4
    bcsUpK = 60;
end
if nargin < 3
    bcsLowK = 2;
end
if nargin < 2
    m = 100; %Ensemble Size
end
if nargin < 1
    dataName = 'Adam';
end
rand('state',0)
if ischar(m) 
    m  = str2num(m);
end
if ischar(bcsLowK)
    bcsLowK = str2num(bcsLowK);
end 
if ischar(bcsUpK)
    bcsUpK = str2num(bcsUpK);
end
if ischar(random_seed)
 random_seed = str2num(random_seed);
end
if bcsUpK < bcsLowK
    bcsUpK = bcsLowK;
end
for seed_index=1:length(random_seed)
    gt = [];
    fea = [];
    %random_seed(seed_index)
    load(['../mdata/',dataName,'_seed_',num2str(random_seed(seed_index)),'.mat'])
    label=double(label);
    fea=data;

    gt=label+1;

    [N, d] = size(fea);
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


        % You can set up parameters by yourself.
        Label = BGEC(fea, k, m, bcsLowK, bcsUpK);

        disp('--------------------------------------------------------------');
        nmiScores(runIdx) = computeNMI(Label,gt);
        ariScores(runIdx) = rand_index(Label,gt, 'adjusted');

        disp(['The NMI score at Run ',num2str(runIdx), ': ',num2str(nmiScores(runIdx))]);
        disp(['The ARI score at Run ',num2str(runIdx), ': ',num2str(ariScores(runIdx))]);
        disp('--------------------------------------------------------------');
    end
    MedianNMI(seed_index,1)=max(nmiScores);
    MedianARI(seed_index,1)=max(ariScores);
end
MedianNMILast=median(MedianNMI(1:length(random_seed),1))
MedianARILast=median(MedianARI(1:length(random_seed),1))
end
