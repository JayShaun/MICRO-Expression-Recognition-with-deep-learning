function file=sortObj(file)
for i=1:length(file)
    A{i}=file(i).name;
end
[~, ind]=natsortfiles(A);
for j=1:length(file)
    files(j)=file(ind(j));
end
clear file;
file=files';