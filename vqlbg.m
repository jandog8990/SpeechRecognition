function c = vqlbg(d, k)
% VQLBG Vector quantization using the Linde-Buzo-Gray algorithm
%
% Inputs:
%       d contains training data vectors (one per column)
%       k is number of centroids required
%
% Outputs:
%       c contains the result VQ codebook (k columns, one for each centroids)

e = 0.01;
dist_thresh = 5.5;
q=1;
%computes the initial centroid of all the acoustic vectors
centr = mean(d,2);
codebook = [centr];

while  size(codebook,2)<k
    
split = zeros(size(d , 1) , size(codebook,2)*2);
for i = 1 : size(codebook,2)
   %splits the codebook; for each codeword currently in the book, it splits
   %it into two (doubling the size of the codebook)
    split(: , 2*i-1:2*i ) = [codebook(:,i)-e , codebook(:,i)+e];
    
end
    codebook = split;
    avg_dist = 10;
    while avg_dist > dist_thresh
        z = disteu(d , codebook);
        %finds the closest codeword in codebook1 to each acoustic vector, where
        %the vector output, j=ind(i) tells that acoustic vector i is closes to
        %codework j
        [m , ind] = min(z , [] , 2);
        avg_dist = mean(m);
        indices = unique(ind);

        %finds all the acoustic vectors associated with code word j, and thus
        %are all part of the same cluster
        codebook = zeros(size(d , 1) , size(indices,1));
        for j = 1 : size(indices,1)
           %seperates the clusters and find the new centroids for the updates
           %the code book
           clust = d(:, find(ind == indices(j) ));
           codebook(: , j) = mean(clust,2); 

           if avg_dist < dist_thresh
                subplot(2,2,q)
                plot(clust(1,:), clust(2,:),'o')
                hold on
                plot(codebook(1,j), codebook(2,j),'o','LineWidth',3)
           end
        end
    end 
   q
   q=q+1;
end
c = codebook;
figure(2)
plot(d(1,:),d(2,:),'o')
hold on 
plot(c(1,:) , c(2,:), 'o','LineWidth',3)

end
