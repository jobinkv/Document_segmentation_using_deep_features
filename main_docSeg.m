function main_docSeg(dataset)
% this super pixel based semantic segmentation 
% This is a testing function
% inputs are : 1) pretrained CNN, 2) Input image, 3) trained SVM model, 4) encoder if any (in dcnn)
% outputs is 1) segmented image
% document image patches. It calculates deep-features aswell as hand crafted features 
%====================================================================================
			%%%%%% Input parameters %%%%%%%%%%
%====================================================================================
setup
global allowdedRatio
dataHomeLoc = ['/users/jobinkv/gt_Hw_dataset_v1/',dataset,'/']; % data path
datafolder = strsplit(dataHomeLoc,'/');
imdb.modelDir='/users/jobinkv/models/'; % model path
imdb.tempModels = ['/users/jobinkv/',datafolder{5},'/'];
imdb.pretrainedNet ='imagenet-vgg-m.mat';% 'net-epoch-5.mat';% 'imagenet-vgg-m.mat';%'imagenet-vgg-verydeep-19.mat';
imdb.shuffle=false; %shuffle image patches across images
allowdedRatio = 0.9;
imdb.patchSize=224;
imdb.stride =100;
imdb.feature = 'rcnn'; % {'dcnn', 'rcnn', 'hog', 'gabor'}
imdb.expDir = ['/tmp/jobinOpts/',datafolder{5},'/',imdb.feature];
opts.useGpu = true;
imdb.imageScale = 4;% resize to 1/imageScale;
imdb.regionSize = 10;
imdb.regularizer = 0.01;
imdb.patchSize = 28;
%-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=
%====================================================================================
			%%%%%% Internal setups  %%%%%%%%%%
%====================================================================================
if (exist(['/tmp/',datafolder{5}])~=7)
        disp('copying the dataset to the /tmp folder ...')
        disp('please wait ...')
        command = ['cp -r ',dataHomeLoc,' /tmp/'];
        [status,cmdout] = system(command,'-echo');
        disp('copyed!')
end

global allowdedRatio
imdb.dataLoc =['/tmp/',datafolder{5}];
imdb = get_database(imdb);
imdb.resultPath = fullfile(imdb.expDir, sprintf('results-%s_stride%d_pZ%d.mat', imdb.feature,imdb.stride,imdb.patchSize)) ;
imdb.encoderPath = fullfile(imdb.modelDir, sprintf('encoder-%s_stride%d_pZ%d.mat', imdb.feature,imdb.stride,imdb.patchSize)) ;
fprintf('Your output will appear at %s\n',imdb.expDir);
fprintf('The data path location is %s\n',imdb.dataLoc);
imdb.encoder=[];
imdb.net = [];
dbstop if error
opts.model=[imdb.modelDir,imdb.pretrainedNet];
switch imdb.pretrainedNet
case {'imagenet-vgg-m.mat'}
opts.layer.rcnn = 19;
opts.layer.dcnn = 13;
case {'imagenet-vgg-verydeep-19.mat'}
opts.layer.dcnn = 39;
end
if exist(imdb.resultPath)
    load(imdb.resultPath)
	disp(['The current setup is already computed please check folder ',imdb.expDir])
	fprintf('mAP train: %.1f, test: %.1f\n', ...
  	mean(info.train.ap)*100, ...
  	mean(info.test.ap)*100);
	return;
end
%====================================================================================
			%%%%%% Load the network and encoder %%%%%%%%%%
%====================================================================================
if (exist('net')~=1)
	switch imdb.feature
  		case {'rcnn'}
    			imdb.net = load(opts.model) ;
    			imdb.net.layers = imdb.net.layers(1:opts.layer.rcnn) ;
    			if opts.useGpu
      				imdb.net = vl_simplenn_move(imdb.net, 'gpu') ;
      				imdb.net.useGpu = true ;
    			else
      				imdb.net = vl_simplenn_move(imdb.net, 'cpu') ;
      				imdb.net.useGpu = false ;
    			end
			imdb.encoder = 'not required';
	    		load(['/users/jobinkv/CSG863_models/rcnn/results-rcnn_scale4_pZ28.mat'],'info');
	    		imdb.classifier.svm.w = info.w;
	    		imdb.classifier.svm.b = info.b;
	    		clear info;
	    		clear net;
  		case {'dcnn'}
    			imdb.net = load(opts.model) ;
    			imdb.net.layers = imdb.net.layers(1:opts.layer.dcnn) ;
    			if opts.useGpu
      				imdb.net = vl_simplenn_move(imdb.net, 'gpu') ;
      				imdb.net.useGpu = true ;
    			else
      				imdb.net = vl_simplenn_move(imdb.net, 'cpu') ;
      				imdb.net.useGpu = false ;
    			end
            		%load(imdb.encoderPath,'encoder');
	    		load(['/users/jobinkv/sentgall_models/dcnn/encoder-dcnn_scale4_pZ28.mat'],'encoder');
            		imdb.encoder = encoder;
	    		load(['/users/jobinkv/sentgall_models/dcnn/results-dcnn_scale4_pZ28.mat'],'info');
	    		imdb.encoder = encoder;
	    		imdb.classifier.svm.w = info.w;
	    		imdb.classifier.svm.b = info.b;
	    		clear info;
			clear encoder;
        	case 'gabor'% this functionality is not ready
            		net = gaborFilterBank(5,8,39,39);
        	case 'fine'% this functionality is not ready
            		net = load(opts.model);
			imdb.net = net.net;
			clear net;
    			if opts.useGpu
      				imdb.net = vl_simplenn_move(imdb.net, 'gpu') ;
      				imdb.net.useGpu = true ;
    			else
      				imdb.net = vl_simplenn_move(imdb.net, 'cpu') ;
      				imdb.net.useGpu = false ;
    			end
			imdb.net.layers{end}.type = 'softmax';
       		case 'dsift'% this functionality is not ready
	    		if (exist(imdb.encoderPath, 'file') == 2)
	    			load(imdb.encoderPath, 'encoder');
            			imdb.encoder = encoder;
			end
    end
end
if (exist(imdb.expDir)~=7)
    mkdir(imdb.expDir)
end
%====================================================================================
for k = 1:numel(imdb.images.name)
    tic
    if (imdb.images.set(k)==1)
	
    patch_result = docseg(imdb,k);
    seg_out = createImg(patch_result);
    imwrite(seg_out,[imdb.expDir,'/',imdb.images.name{k}]);
    end
    toc
end
fprintf('Your output will appear at %s\n',imdb.expDir);
%--------------------------------------------------
function [out_image] = createImg(patch_result)
out_image = zeros(size(patch_result.segments,1),size(patch_result.segments,2),3);
for i =1:numel(patch_result.results)
	[val,loc] = max(patch_result.results{i}.probs);
		if (loc>1)
			[row,col,v] = find(patch_result.segments==patch_result.results{i}.segId);
			for i=1:size(row,1)
 				out_image(row(i),col(i),loc-1) = 255;
   			end 
		end
end
function [out_image] = createImg_dummy(patch_result)
out_image = zeros(size(patch_result.segments));
for i =1:numel(patch_result.results)
	[val,loc] = max(patch_result.results{i}.probs);
	vals=zeros(3,1);
	vals(loc,1)=255;
	red=vals(1,1);
	green=vals(2,1);
	blue=vals(3,1);
	%red=ceil((patch_result.results{i}.probs(1)+1)*255/2)
	%green=ceil((patch_result.results{i}.probs(2)+1)*255/2)
	%blue=ceil((patch_result.results{i}.probs(3)+1)*255/2)
	out_image(patch_result.results{i}.loc.y-imdb.stride/2:patch_result.results{i}.loc.y+imdb.stride/2,patch_result.results{i}.loc.x-imdb.stride/2:patch_result.results{i}.loc.x+imdb.stride/2,1)=red;
	out_image(patch_result.results{i}.loc.y-imdb.stride/2:patch_result.results{i}.loc.y+imdb.stride/2,patch_result.results{i}.loc.x-imdb.stride/2:patch_result.results{i}.loc.x+imdb.stride/2,2)=green;
	out_image(patch_result.results{i}.loc.y-imdb.stride/2:patch_result.results{i}.loc.y+imdb.stride/2,patch_result.results{i}.loc.x-imdb.stride/2:patch_result.results{i}.loc.x+imdb.stride/2,3)=blue;
end
%--------------------------------------------------
    function [out] = docseg(imdb,ii)      
image1 = imread(fullfile(imdb.imageDir,imdb.images.name{ii}));
image =imresize(image1,1/imdb.imageScale);
% slic
segments = vl_slic(single(image), imdb.regionSize, imdb.regularizer, 'verbose') ;
imdb.no_patches = max(max(segments));
        height = size(image,1);
        width  = size(image,2);
        cnt=0;
        for i=1:imdb.no_patches;
	    [row,col,v] = find(segments==i);
	    mean_row = floor(mean(row));
	    mean_col = floor(mean(col));
	    y_min = mean_row - imdb.patchSize/2;
	    x_min = mean_col - imdb.patchSize/2;
	    if (x_min>0 && y_min>0 && y_min+imdb.patchSize<height && x_min+imdb.patchSize<width)
    	    cnt=cnt+1;
    	    %img_patchs{cnt}.patch =imresize(tmpimg,[224 224]);% tmpimg;
    	    img_patchs{cnt}.patch =imcrop(image,[x_min y_min  imdb.patchSize imdb.patchSize]);
	    img_patchs{cnt}.segId = i;
            %for y=1:imdb.stride:height-imdb.patchSize;
            %    cnt=cnt+1;
            %    img_patchs{cnt}.patch = image(y:y+imdb.patchSize-1, x:x+imdb.patchSize-1);
            %    img_patchs{cnt}.loc.x = x+floor(imdb.patchSize/2);
            %    img_patchs{cnt}.loc.y = y+floor(imdb.patchSize/2);
            end
        end
	clear image;
	clear image1;
	batch_size = 128;
	% get the features in batch
	for j=1:batch_size:numel(img_patchs)
	cnt=1;
	for i=j:j+batch_size
		if (i<=numel(img_patchs))	
		batch_patch{cnt} =img_patchs{i}.patch; 
		else 
		break;
		end
%		batch_loc{i} =img_patchs{i}.loc;
		cnt=cnt+1; 
	end
        switch imdb.feature
            case 'rcnn'
                feature=get_rcnn_features_modified(imdb.net, batch_patch);
            case 'dcnn'
                feature = get_dcnn_features_modified(imdb.net,batch_patch,'encoder',imdb.encoder);
    	    case 'dsift'
        	feature = get_dcnn_features_modified([],batch_patch,'useSIFT', true,'encoder', imdb.encoder,'numSpatialSubdivisions',1,'maxNumLocalDescriptorsReturned', 500);
            case 'gabor'
                feature = get_gabor_feature(imdb.net,batch_patch);
            case 'fine'
                feature = get_fineTurned_patchClass(imdb.net,batch_patch);
        end
	cnt=1;
	for i=j:j+batch_size
		if (i<=numel(img_patchs))	
	%	result{i}.probs = imdb.classifier.svm.w'*feature{cnt}+imdb.classifier.svm.b';
		switch imdb.feature
		case {'fine'}
		result{i}.probs = feature{cnt};
		otherwise
		result{i}.probs = bsxfun(@plus, imdb.classifier.svm.w'*feature{cnt}, imdb.classifier.svm.b');
	 	end
		result{i}.segId = img_patchs{i}.segId;
		cnt=cnt+1;
		else 
		break;
		end
	end
	clear feature;
	end
	out.results = result;
        out.segments = segments;
	clear result;
  % end        
        %code = get_rcnn_features_modified(net, bachGround_patch);

function imdb = get_database(imdb,varargin)
opts.seed = 1 ;
opts = vl_argparse(opts, varargin) ;
imdb.imageDir = fullfile(imdb.dataLoc, 'image') ;
imdb.maskDir = fullfile(imdb.dataLoc, 'gti') ;
imdb.classes.name={...
    	'background'
    	'comment'
    	'decoration'
	'text'
	'text+comment'
	'text+decoration'
	'comment+decoration'};
numClass = length(imdb.classes.name);
imageFiles = dir(fullfile(imdb.imageDir, '*.jpg'));
imdb.images.name = {imageFiles.name};
numImages = length(imdb.images.name);
% finding the number of patches
%image = imread(fullfile(imdb.imageDir,imdb.images.name{1}));
%height = size(image,1);
%width  = size(image,2);
%no_xmoves = floor( (width-imdb.patchSize)/imdb.stride + 1);
%no_ymoves = floor((height-imdb.patchSize)/imdb.stride + 1);
%imdb.no_patches = floor(no_ymoves*no_xmoves);
%imdb.images.label = ones(numClass, numImages*imdb.no_patches)*-1;
imdb.images.vocid = cellfun(@(S) S(1:end-4), imdb.images.name, 'UniformOutput', false);
imdb.images.set = zeros(1, numImages);
imdb.images.id = 1:numImages;
% Loop over images and record the imag sets
imageSets = {'test', 'train', 'valid'};
for s = 1:length(imageSets),
    imageSetPath = fullfile(imdb.dataLoc, 'ImageSet', sprintf('%s.txt',imageSets{s}));
    gtids1 = textread(imageSetPath,'%s');
    gtids = cellfun(@(S) S(1:end-4), gtids1, 'UniformOutput', false);
    [membership, loc] = ismember(gtids, imdb.images.vocid);
    assert(all(membership));
    imdb.images.set(loc) = s;
end
% Remove images not part of train, val, test sets
valid = ismember(imdb.images.set, 1:length(imageSets));
imdb.images.name = imdb.images.name(imdb.images.id(valid));
imdb.images.id = 1:numel(imdb.images.name);
%imdb.images.label = imdb.images.label(:, valid);
imdb.images.set = imdb.images.set(valid);
imdb.images.vocid = imdb.images.vocid(valid);
%------------------------------------------------------
