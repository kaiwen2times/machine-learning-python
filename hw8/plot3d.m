function p=plot3d(x,varargin);
% plot3d: shortcut for plot3(x(:,1),x(:,2),x(:,3),...)
if nargout
   if isempty(varargin)
      p=plot3(x(:,1),x(:,2),x(:,3));
   else
      p=plot3(x(:,1),x(:,2),x(:,3),varargin{:});
   end
else
   if isempty(varargin)
      plot3(x(:,1),x(:,2),x(:,3));
   else
      plot3(x(:,1),x(:,2),x(:,3),varargin{:});
   end
end


