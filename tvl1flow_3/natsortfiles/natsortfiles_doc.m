%% NATSORTFILES Examples
% The function <https://www.mathworks.com/matlabcentral/fileexchange/47434
% |NATSORTFILES|> sorts a cell array of filenames or filepaths, taking into
% account any number values within the strings. This is known as a "natural
% order sort" or an "alphanumeric sort". Note that MATLAB's inbuilt
% <http://www.mathworks.com/help/matlab/ref/sort.html |SORT|> function
% sorts the character codes only (as per |sort| in most programming languages).
%
% |NATSORTFILES| is not a naive natural-order sort, but sorts the filenames
% and file extensions separately: this prevents the file extension separator
% character |.| from influencing the sort order. The effect of this is that
% shorter filenames and directory names always sort before longer ones,
% thus providing a dictionary sort of the names. For the same reason
% filepaths are split at each path separator character, and each directory
% level is sorted separately. See the "Explanation" sections for more details.
%
% For sorting the rows of a cell array of strings use
% <https://www.mathworks.com/matlabcentral/fileexchange/47433 |NATSORTROWS|>.
%
% For sorting a cell array of strings use
% <https://www.mathworks.com/matlabcentral/fileexchange/34464 |NATSORT|>.
%
%% Basic Usage:
% By default |NATSORTFILES| interprets consecutive digits as being part of
% a single integer, each number is considered to be as wide as one letter:
A = {'a2.txt', 'a10.txt', 'a1.txt'};
sort(A)
natsortfiles(A)
%% Output 2: Sort Index
% The second output argument is a numeric array of the sort indices |ndx|,
% such that |Y = X(ndx)| where |Y = natsortfiles(X)|:
[~,ndx] = natsortfiles(A)
%% Explanation: Dictionary Sort
% Filenames and file extensions are separated by the extension separator,
% the period character |.|, which gets sorted _after_ all of the characters
% from 0 to 45, including |!"#$%&'()*+,-|, the space character, and all of
% the control characters (newlines, tabs, etc). This means that a naive
% sort or natural-order sort will sort some short filenames after longer
% filenames. In order to provide the correct dictionary sort, with shorter
% filenames first, |NATSORTFILES| sorts the filenames and file extensions
% separately:
B = {'test_new.m'; 'test-old.m'; 'test.m'};
sort(B) % '-' sorts before '.'
natsort(B) % '-' sorts before '.'
natsortfiles(B) % correct dictionary sort
%% Explanation: Filenames
% |NATSORTFILES| combines a dictionary sort with a natural-order sort, so
% that the number values within the filenames are taken into consideration:
C = {'test2.m'; 'test10-old.m'; 'test.m'; 'test10.m'; 'test1.m'};
sort(C) % Wrong numeric order.
natsort(C) % Correct numeric order, but longer before shorter.
natsortfiles(C) % Correct numeric order and dictionary sort.
%% Explanation: Filepaths
% For the same reason, filepaths are split at each file path separator
% character (both |/| and |\| are considered to be file path separators)
% and every level of directory names are sorted separately. This ensures
% that the directory names are sorted with a dictionary sort and that any
% numbers are taken into consideration:
D = {'A2-old\test.m';'A10\test.m';'A2\test.m';'A1archive.zip';'A1\test.m'};
sort(D) % Wrong numeric order, and '-' sorts before '\':
natsort(D) % correct numeric order, but longer before shorter.
natsortfiles(D) % correct numeric order and dictionary sort.
%% Regular Expression: Decimal Numbers, E-notation, +/- Sign.
% |NATSORTFILES| is a wrapper for |NATSORT|, which means all of |NATSORT|'s
% options are also supported. In particular the number recognition can be
% customized to detect numbers with decimal digits, E-notation, a +/- sign,
% or other specific features. This detection is defined by providing an
% appropriate regular expression: see |NATSORT| for details and examples.
E = {'test24.csv','test1.2.csv','test5.csv','test3.3.csv','test12.csv'};
natsort(E,'\d+(\.\d+)?')