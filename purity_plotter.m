clear, clc

generic_path = '/Users/bcs/Desktop/Repos/MSc/bcs_tracker/dukemtmc/purity_results%s/camera%d/tl%d_iou%0.5g_d%0.5g.txt';

% ious = [0.95, 0.9, 0.75, 0.6, 0.5];
% cos_dists = [0.5, 0.3, 0.1, 0.05, 0.005];

tracklet_lengths = [5, 10, 15, 20, 30, 40, 50];
ious = [0.95, 0.75, 0.5];
cos_dists = [0.3, 0.1, 0.05];


cos_axis_length = size(cos_dists, 2);
iou_axis_length = size(ious, 2);


for cam_id = 1:8
figure('NumberTitle', 'off', 'Name', sprintf('Camera %d', cam_id));

for ioui = 1:iou_axis_length
    for ci = 1:cos_axis_length
        subplot_id = (ioui-1)*cos_axis_length + ci;
        subplot(iou_axis_length, cos_axis_length, subplot_id);
        purities = zeros(size(tracklet_lengths));
        
        for ti = 1:size(tracklet_lengths, 2)
            clear A
            current_path = sprintf(generic_path, '_new', cam_id, tracklet_lengths(ti), ious(ioui), cos_dists(ci));
            A = load(current_path);
            purities(ti) = sum(A(:, 2) > 0.99)/size(A,1);
            
        end
        
        plot(tracklet_lengths, purities, 'LineWidth', 2)
        grid on
        plot_title = sprintf('iou: %.6g, d: %.6g', ious(ioui), cos_dists(ci));
        title(plot_title)
        xlim([5 50])
        xlabel('Tracklet Length')
        ylabel('Purity')
    end
end

end

% sgtitle(sprintf('Camera %d', cam_id))

