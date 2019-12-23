clear, clc



figure('NumberTitle', 'off', 'Name', 'Tracklet Length = 10');
subplot(3,5,1)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.75_d0.2.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.7, d=0.2, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,2)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.75_d0.1.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.7, d=0.1, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,3)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.75_d0.05.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.7, d=0.05, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,4)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.75_d0.01.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.7, d=0.01, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,5)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.75_d0.005.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.7, d=0.005, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)


subplot(3,5,6)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.6_d0.2.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.6, d=0.2, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,7)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.6_d0.1.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.6, d=0.1, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,8)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.6_d0.05.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.6, d=0.05, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,9)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.6_d0.01.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.6, d=0.01, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,10)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.6_d0.005.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.6, d=0.005, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)



subplot(3,5,11)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.5_d0.2.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.5, d=0.2, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,12)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.5_d0.1.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.5, d=0.1, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,13)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.5_d0.05.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.5, d=0.05, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,14)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.5_d0.01.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.5, d=0.01, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,15)
A = load('dukemtmc/purity_results/camera3_small/tl10_iou0.5_d0.005.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.5, d=0.005, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)



figure('NumberTitle', 'off', 'Name', 'Tracklet Length = 5');
subplot(3,5,1)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.75_d0.2.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.7, d=0.2, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,2)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.75_d0.1.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.7, d=0.1, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,3)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.75_d0.05.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.7, d=0.05, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,4)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.75_d0.01.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.7, d=0.01, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,5)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.75_d0.005.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.7, d=0.005, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)


subplot(3,5,6)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.6_d0.2.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.6, d=0.2, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,7)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.6_d0.1.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.6, d=0.1, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,8)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.6_d0.05.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.6, d=0.05, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,9)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.6_d0.01.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.6, d=0.01, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,10)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.6_d0.005.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.6, d=0.005, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)



subplot(3,5,11)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.5_d0.2.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.5, d=0.2, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,12)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.5_d0.1.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.5, d=0.1, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,13)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.5_d0.05.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.5, d=0.05, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,14)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.5_d0.01.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.5, d=0.01, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)
subplot(3,5,15)
A = load('dukemtmc/purity_results/camera3_small/tl5_iou0.5_d0.005.txt');
binary_purity = sum(A(:, 2)==1.0)/size(A,1);
histogram(A(:, 2))
str = sprintf('iou=0.5, d=0.005, purity=%.4f', binary_purity);
title(str)
str = sprintf('%d/%d', sum(A(:, 2)==1.0),size(A,1));
xlabel(str)