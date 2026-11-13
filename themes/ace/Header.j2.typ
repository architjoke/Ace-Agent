((* if cv.photo *))
#two-col(
  left-column-width: design-header-photo-width * 1.1,
  right-column-width: 1fr,
  left-content: [
    #align(
      left + horizon,
      image("profile_picture.jpg", width: design-header-photo-width),
    )
  ],
  column-gutter: 0cm,
  right-content: [
((* endif *))
((* if cv.name *))
= <<cv.name|escape_typst_characters>>
((* endif *))

// Compact connections: icon + value only (no Location:/Email: labels — those
// force 2–3 wrapped header rows and leave the bottom of the page empty).
#let connections-list = (
((* for connection in cv.connections *))
  [((*- if connection["url"] and design.header.make_connections_links -*))
  #box(original-link("<<connection["url"]>>")[
  ((*- endif -*))
  ((*- if design.header.use_icons_for_connections -*))
    #fa-icon("<<connection["typst_icon"]>>", size: 0.85em) #h(0.06cm)
  ((*- endif -*))
  ((*- if connection["typst_icon"] in ["linkedin", "github", "link"] and connection["url"] -*))
    <<connection["clean_url"]|escape_typst_characters>>
  ((*- else -*))
    <<connection["placeholder"]|escape_typst_characters>>
  ((*- endif -*))
  ((*- if connection["url"] and design.header.make_connections_links -*))
  ])
  ((*- endif -*))],
((* endfor *))
)
#connections(connections-list)

((* if cv.photo *))
  ],
)
((* endif *))
