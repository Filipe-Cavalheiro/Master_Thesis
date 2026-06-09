classdef BinaryMinHeap < handle
    % BinaryMinHeap for (key, value) pairs where key = distance, value = vertex id.
    % Supports push and popMin. Uses "lazy decrease-key": just push new (smaller) keys.
    properties
        keys   (:,1) double = zeros(0,1)
        vals   (:,1) double = zeros(0,1)
    end

    methods
        function n = size(h)
            n = numel(h.keys);
        end

        function tf = isEmpty(h)
            tf = isempty(h.keys);
        end

        function push(h, key, val)
            h.keys(end+1,1) = key;
            h.vals(end+1,1) = val;
            h.siftUp(numel(h.keys));
        end

        function [key, val] = popMin(h)
            if isempty(h.keys)
                error('BinaryMinHeap:Empty', 'Heap is empty.');
            end
            key = h.keys(1);
            val = h.vals(1);

            last = numel(h.keys);
            if last == 1
                h.keys = zeros(0,1);
                h.vals = zeros(0,1);
                return;
            end

            h.keys(1) = h.keys(last);
            h.vals(1) = h.vals(last);
            h.keys(last) = [];
            h.vals(last) = [];
            h.siftDown(1);
        end
    end

    methods (Access = private)
        function siftUp(h, i)
            while i > 1
                p = floor(i/2);
                if h.keys(p) <= h.keys(i)
                    break;
                end
                h.swap(i, p);
                i = p;
            end
        end

        function siftDown(h, i)
            n = numel(h.keys);
            while true
                l = 2*i;
                r = l + 1;
                if l > n, break; end

                m = l;
                if r <= n && h.keys(r) < h.keys(l)
                    m = r;
                end

                if h.keys(i) <= h.keys(m)
                    break;
                end

                h.swap(i, m);
                i = m;
            end
        end

        function swap(h, i, j)
            tk = h.keys(i); h.keys(i) = h.keys(j); h.keys(j) = tk;
            tv = h.vals(i); h.vals(i) = h.vals(j); h.vals(j) = tv;
        end
    end
end