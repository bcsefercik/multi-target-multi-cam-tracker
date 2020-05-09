clear, clc

% generic_path = '/Users/bcs/Desktop/Repos/MSc/bcs_tracker/dukemtmc/purity_results%s/camera%d/tl%d_iou%0.5g_d%0.5g.txt';
generic_path = '/Users/bcs/Desktop/Repos/MSc/purity_results/openpose_60fps_val%s/camera%d/tl%d_iou%0.5g_d%0.5g.txt';

% ious = [0.95, 0.9, 0.75, 0.6, 0.5];
% cos_dists = [0.5, 0.3, 0.1, 0.05, 0.005];

% tracklet_lengths = [5, 10, 15, 20, 30, 40, 50];
tracklet_lengths = [5 10 15 20 30 60 120];
ious = [0.75, 0.85];
cos_dists = [0.3 0.4];


cos_axis_length = size(cos_dists, 2);
iou_axis_length = size(ious, 2);


for camid = 1:8
% for camid = [1 2 3 5 6 7 8]
figure('NumberTitle', 'off', 'Name', sprintf('Camera %d', camid));

for ioui = 1:iou_axis_length
    for ci = 1:cos_axis_length
        subplot_id = (ioui-1)*cos_axis_length + ci;
        subplot(iou_axis_length, cos_axis_length, subplot_id);
%         subplot_id = (ci-1)*iou_axis_length + ioui;
%         subplot(cos_axis_length, iou_axis_length, subplot_id);
        purities = zeros(size(tracklet_lengths));
        
        for ti = 1:size(tracklet_lengths, 2)
            clear A
            current_path = sprintf(generic_path, '', camid, tracklet_lengths(ti), ious(ioui), cos_dists(ci));
            
            A = load(current_path);
            purities(ti) = sum(A(:, 2) > 0.99)/size(A,1);
            
        end
        
        plot([tracklet_lengths(1:end-1), inf], purities, 'LineWidth', 2)
        grid on
        plot_title = sprintf('iou: %.6g, d: %.6g', ious(ioui), cos_dists(ci));
        title(plot_title)
        xlim([min(tracklet_lengths) inf])
        ylim([min(purities)-1e-6 1.0+1e-6])
        xlabel('Tracklet Length')
        ylabel('Purity')
    end
end

end

% sgtitle(sprintf('Camera %d', camid))

