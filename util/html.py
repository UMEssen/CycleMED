import os
from html import escape

try:
    import dominate
    from dominate.tags import meta, h3, table, tr, td, p, a, img, br
except ImportError:
    dominate = None


class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.

     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes

        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.headers = []
        self.image_rows = []

        if dominate is not None:
            self.doc = dominate.document(title=title)
            if refresh > 0:
                with self.doc.head:
                    meta(http_equiv="refresh", content=str(refresh))
        else:
            self.doc = None
            self.refresh = refresh

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file

        Parameters:
            text (str) -- the header text
        """
        if self.doc is not None:
            with self.doc:
                h3(text)
        else:
            self.headers.append(text)

    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        if self.doc is not None:
            self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
            self.doc.add(self.t)
            with self.t:
                with tr():
                    for im, txt, link in zip(ims, txts, links):
                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with p():
                                with a(href=os.path.join('images', link)):
                                    img(style="width:%dpx" % width, src=os.path.join('images', im))
                                br()
                                p(txt)
        else:
            self.image_rows.append((ims, txts, links, width))

    def save(self):
        """save the current content to the HMTL file"""
        html_file = '%s/index.html' % self.web_dir
        with open(html_file, 'wt') as f:
            if self.doc is not None:
                f.write(self.doc.render())
                return

            refresh = ''
            if self.refresh > 0:
                refresh = f'<meta http-equiv="refresh" content="{int(self.refresh)}">\n'
            f.write('<!doctype html>\n<html><head>\n')
            f.write(f'<meta charset="utf-8">\n{refresh}<title>{escape(self.title)}</title>\n')
            f.write('</head><body>\n')
            for header in self.headers:
                f.write(f'<h3>{escape(header)}</h3>\n')
            for ims, txts, links, width in self.image_rows:
                f.write('<table border="1" style="table-layout: fixed;"><tr>\n')
                for im, txt, link in zip(ims, txts, links):
                    image_src = escape(os.path.join('images', im))
                    href = escape(os.path.join('images', link))
                    f.write('<td style="word-wrap: break-word;" align="center" valign="top">')
                    f.write(f'<p><a href="{href}"><img style="width:{int(width)}px" src="{image_src}"></a><br>')
                    f.write(f'{escape(txt)}</p></td>\n')
                f.write('</tr></table>\n')
            f.write('</body></html>\n')


if __name__ == '__main__':  # we show an example usage here.
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims, txts, links = [], [], []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
