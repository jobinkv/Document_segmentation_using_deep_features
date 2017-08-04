function info = traintest_modified(imdb, psi)
% -------------------------------------------------------------------------
disp('in traintest_modified');
multiLabel = (size(imdb.segments.label,1) > 1) ;
% print some data statistics

train = ismember(imdb.segments.set, [1 2 3]) ;
test = ismember(imdb.segments.set, 1) ;

info.classes = [1 1 1 1] ;
C = 1 ;
w = {} ;
b = {} ;

for c=1:numel(info.classes)
  if ~multiLabel
    y = 2*(imdb.segments.label == info.classes(c)) - 1 ;
  else
    y = imdb.segments.label(c,:) ;
  end
  np = sum(y(train) > 0) ;
  nn = sum(y(train) < 0) ;
  n = np + nn ;

  [w{c},b{c}] = vl_svmtrain(psi(:,train & y ~= 0), y(train & y ~= 0), 1/(n* C), ...
    'epsilon', 0.001, 'verbose', 'biasMultiplier', 1, ...
    'maxNumIterations', n * 200) ;

  pred = w{c}'*psi + b{c} ;

  % try cheap calibration
  mp = median(pred(train & y > 0)) ;
  mn = median(pred(train & y < 0)) ;
  b{c} = (b{c} - mn) / (mp - mn) ;
  w{c} = w{c} / (mp - mn) ;
  pred = w{c}'*psi + b{c} ;
  scores{c} = pred ;
  [~,~,i]= vl_pr(y(train), pred(train)) ; ap(c) = i.ap ; ap11(c) = i.ap_interp_11 ;
  [~,~,i]= vl_pr(y(test), pred(test)) ; tap(c) = i.ap ; tap11(c) = i.ap_interp_11 ;
  [~,~,i]= vl_pr(y(train), pred(train), 'normalizeprior', 0.01) ; nap(c) = i.ap ;
  [~,~,i]= vl_pr(y(test), pred(test), 'normalizeprior', 0.01) ; tnap(c) = i.ap ;
  
end
info.w = cat(2,w{:}) ;
info.b = cat(2,b{:}) ;
info.scores = cat(1, scores{:}) ;
info.train.ap = ap ;
info.train.ap11 = ap11 ;
info.train.nap = nap ;
info.train.map = mean(ap) ;
info.train.map11 = mean(ap11) ;
info.train.mnap = mean(nap) ;
info.test.ap = tap ;
info.test.ap11 = tap11 ;
info.test.nap = tnap ;
info.test.map = mean(tap) ;
info.test.map11 = mean(tap11) ;
info.test.mnap = mean(tnap) ;
clear ap nap tap tnap scores ;
fprintf('mAP train: %.1f, test: %.1f\n', ...
  mean(info.train.ap)*100, ...
  mean(info.test.ap)*100);

figure(1) ; clf ;
subplot(3,2,1) ;
bar([info.train.ap; info.test.ap]')
xlabel('class') ;
ylabel('AP') ;
legend(...
  sprintf('train (%.1f)', info.train.map*100), ...
  sprintf('test (%.1f)', info.test.map*100));
title('average precision') ;

subplot(3,2,2) ;
bar([info.train.nap; info.test.nap]')
xlabel('class') ;
ylabel('AP') ;
legend(...
  sprintf('train (%.1f)', info.train.mnap*100), ...
  sprintf('test (%.1f)', info.test.mnap*100));
title('normalized average precision') ;

if ~multiLabel
  [~,preds] = max(info.scores,[],1) ;
  [~,gts] = ismember(imdb.segments.label, info.classes) ;

%   % per pixel
%   [info.train.msrcConfusion, info.train.msrcAcc] = compute_confusion(numel(info.classes), gts(train), preds(train), imdb.segments.area(train), true) ;
%   [info.test.msrcConfusion, info.test.msrcAcc] = compute_confusion(numel(info.classes), gts(test), preds(test), imdb.segments.area(test), true) ;
% 
%   % per pixel per class
%   [info.train.confusion, info.train.acc] = compute_confusion(numel(info.classes), gts(train), preds(train), imdb.segments.area(train)) ;
%   [info.test.confusion, info.test.acc] = compute_confusion(numel(info.classes), gts(test), preds(test), imdb.segments.area(test)) ;

  % per segment per class
  [info.train.psConfusion, info.train.psAcc] = compute_confusion(numel(info.classes), gts(train), preds(train)) ;
  [info.test.psConfusion, info.test.psAcc] = compute_confusion(numel(info.classes), gts(test), preds(test)) ;

  subplot(3,2,3) ;
  imagesc(info.train.confusion) ;
  title(sprintf('train confusion per pixel (acc: %.1f, msrc acc: %.1f)', ...
    info.train.acc*100, info.train.msrcAcc*100)) ;

  subplot(3,2,4) ;
  imagesc(info.test.confusion) ;
  title(sprintf('test confusion per pixel (acc: %.1f, msrc acc: %.1f)', ...
    info.test.acc*100, info.test.msrcAcc*100)) ;

  subplot(3,2,5) ;
  imagesc(info.train.psConfusion) ;
  title(sprintf('train confusion per segment (acc: %.1f)', info.train.psAcc*100)) ;

  subplot(3,2,6) ;
  imagesc(info.test.psConfusion) ;
  title(sprintf('test confusion per segment (acc: %.1f)', info.test.psAcc*100)) ;
end
