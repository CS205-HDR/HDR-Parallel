def blend(im1, im2, alpha):
    """
    Creates a new image by interpolating between two input images, using
    a constant alpha.::

        out = image1 * (1.0 - alpha) + image2 * alpha

    :param im1: The first image.
    :param im2: The second image.  Must have the same mode and size as
       the first image.
    :param alpha: The interpolation alpha factor.  If alpha is 0.0, a
       copy of the first image is returned. If alpha is 1.0, a copy of
       the second image is returned. There are no restrictions on the
       alpha value. If necessary, the result is clipped to fit into
       the allowed output range.
    :returns: An :py:class:`~PIL.Image.Image` object.
    """

    im1.load()
    im2.load()
    return im1._new(core.blend(im1.im, im2.im, alpha))

def load(self):
        """
        Allocates storage for the image and loads the pixel data.  In
        normal cases, you don't need to call this method, since the
        Image class automatically loads an opened image when it is
        accessed for the first time. This method will close the file
        associated with the image.

        :returns: An image access object.
        :rtype: :ref:`PixelAccess` or :py:class:`PIL.PyAccess`
        """
        if self.im and self.palette and self.palette.dirty:
            # realize palette
            self.im.putpalette(*self.palette.getdata())
            self.palette.dirty = 0
            self.palette.mode = "RGB"
            self.palette.rawmode = None
            if "transparency" in self.info:
                if isinstance(self.info["transparency"], int):
                    self.im.putpalettealpha(self.info["transparency"], 0)
                else:
                    self.im.putpalettealphas(self.info["transparency"])
                self.palette.mode = "RGBA"

        if self.im:
            if HAS_CFFI and USE_CFFI_ACCESS:
                if self.pyaccess:
                    return self.pyaccess
                from PIL import PyAccess
                self.pyaccess = PyAccess.new(self, self.readonly)
                if self.pyaccess:
                    return self.pyaccess
            return self.im.pixel_access(self.readonly)

def _new(self, im):
        new = Image()
        new.im = im
        new.mode = im.mode
        new.size = im.size
        new.palette = self.palette
        if im.mode == "P" and not new.palette:
            from PIL import ImagePalette
            new.palette = ImagePalette.ImagePalette()
        try:
            new.info = self.info.copy()
        except AttributeError:
            # fallback (pre-1.5.2)
            new.info = {}
            for k, v in self.info:
                new.info[k] = v
        return new

def composite(image1, image2, mask):
    """
    Create composite image by blending images using a transparency mask.

    :param image1: The first image.
    :param image2: The second image.  Must have the same mode and
       size as the first image.
    :param mask: A mask image.  This image can have mode
       "1", "L", or "RGBA", and must have the same size as the
       other two images.
    """

    image = image2.copy()
    image.paste(image1, None, mask)
    return image

def paste(self, im, box=None, mask=None):
        """
        Pastes another image into this image. The box argument is either
        a 2-tuple giving the upper left corner, a 4-tuple defining the
        left, upper, right, and lower pixel coordinate, or None (same as
        (0, 0)).  If a 4-tuple is given, the size of the pasted image
        must match the size of the region.

        If the modes don't match, the pasted image is converted to the mode of
        this image (see the :py:meth:`~PIL.Image.Image.convert` method for
        details).

        Instead of an image, the source can be a integer or tuple
        containing pixel values.  The method then fills the region
        with the given color.  When creating RGB images, you can
        also use color strings as supported by the ImageColor module.

        If a mask is given, this method updates only the regions
        indicated by the mask.  You can use either "1", "L" or "RGBA"
        images (in the latter case, the alpha band is used as mask).
        Where the mask is 255, the given image is copied as is.  Where
        the mask is 0, the current value is preserved.  Intermediate
        values can be used for transparency effects.

        Note that if you paste an "RGBA" image, the alpha band is
        ignored.  You can work around this by using the same image as
        both source image and mask.

        :param im: Source image or pixel value (integer or tuple).
        :param box: An optional 4-tuple giving the region to paste into.
           If a 2-tuple is used instead, it's treated as the upper left
           corner.  If omitted or None, the source is pasted into the
           upper left corner.

           If an image is given as the second argument and there is no
           third, the box defaults to (0, 0), and the second argument
           is interpreted as a mask image.
        :param mask: An optional mask image.
        """

        if isImageType(box) and mask is None:
            # abbreviated paste(im, mask) syntax
            mask = box
            box = None

        if box is None:
            # cover all of self
            box = (0, 0) + self.size

        if len(box) == 2:
            # lower left corner given; get size from image or mask
            if isImageType(im):
                size = im.size
            elif isImageType(mask):
                size = mask.size
            else:
                # FIXME: use self.size here?
                raise ValueError(
                    "cannot determine region size; use 4-item box"
                    )
            box = box + (box[0]+size[0], box[1]+size[1])

        if isStringType(im):
            from PIL import ImageColor
            im = ImageColor.getcolor(im, self.mode)

        elif isImageType(im):
            im.load()
            if self.mode != im.mode:
                if self.mode != "RGB" or im.mode not in ("RGBA", "RGBa"):
                    # should use an adapter for this!
                    im = im.convert(self.mode)
            im = im.im

        self.load()
        if self.readonly:
            self._copy()

        if mask:
            mask.load()
            self.im.paste(im, box, mask.im)
        else:
            self.im.paste(im, box)
