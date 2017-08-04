function code = get_rcnn_features_modified(net, im)
% GET_RCNN_FEATURES
%    This function gets the fc7 features for an image region,
%    extracted from the provided mask.
opts.regionBorder = 0.05 ;
opts.batchSize = 96 ;

if ~iscell(im)
  im = {im} ;
end

res = [] ;
cache = struct() ;
resetCache() ;

% for each image
  function resetCache()
    cache.images = cell(1,opts.batchSize) ;
    cache.indexes = zeros(2, opts.batchSize) ;
    cache.numCached = 0 ;
  end

  function flushCache()
    if cache.numCached == 0, return ; end
    images = cat(4, cache.images{:}) ;
    images = bsxfun(@minus, images, net.meta.normalization.averageImage) ;
    if net.useGpu
      images = gpuArray(images) ;
    end
    res = vl_simplenn(net, images, ...
      [], res, ...
      'conserveMemory', true, ...
      'sync', true) ;
    code_ = squeeze(gather(res(end).x)) ;
    code_ = bsxfun(@times, 1./sqrt(sum(code_.^2)), code_) ;
    for q=1:cache.numCached
      code{cache.indexes(1,q)}{cache.indexes(2,q)} = code_(:,q) ;
    end
    resetCache() ;
  end

  function appendCache(i,r,im)
    cache.numCached = cache.numCached + 1 ;
    cache.images{cache.numCached} = im ;
    cache.indexes(:,cache.numCached) = [i ; r] ;
    if cache.numCached >= opts.batchSize
      flushCache() ;
    end
  end

code = {} ;
for k=1:numel(im)

appendCache(k, 1, ...
      getRegion(single(im{k}), net.meta.normalization.imageSize(1))) ;
    if 0
      figure(1) ; clf ;
      subplot(2,2,1) ; imagesc(im{k}) ;
      drawnow ;
    end

end
flushCache() ;

for k=1:numel(code)
  code{k} = cat(2, code{k}{:}) ;
end
end

% -------------------------------------------------------------------------
function reg = getRegion(im,regionSize)
% -------------------------------------------------------------------------
reg = im;
% resize it
reg = imresize(reg, [regionSize, regionSize], 'bicubic') ;

end

% -------------------------------------------------------------------------
function box = insideBox(mask)
% -------------------------------------------------------------------------
mask_ = pad2(single(mask(1:2:end,1:2:end)), 50, 50, 50, 50) ;

[frames,~,info] = vl_covdet(mask_,'DoubleImage',false) ;
frames = frames(:, info.peakScores > 0)*2  ;

if isempty(frames)
  box = enclosingBox(mask) ;
  return;
end

boxes = [...
  frames(1:2,:) - frames([3 6],:) - 100;
  frames(1:2,:) + frames([3 6],:) - 100] ;

area = boxarea(boxes) ;
[~,best] = max(boxarea(boxes)) ;
box = boxes(:,best) ;

%figure(101) ; clf ; imagesc(mask);axis equal ;hold on ;vl_plotbox(boxes) ;
%vl_plotbox(box,'linewidth', 4) ; drawnow ;
end

% -------------------------------------------------------------------------
function box = enclosingBox(mask)
% -------------------------------------------------------------------------
[x,y] = meshgrid(1:size(mask,2), 1:size(mask,1)) ;
x = x(mask) ;
y = y(mask) ;
box = [min(x) ; min(y) ; max(x) ; max(y)] ;
end
