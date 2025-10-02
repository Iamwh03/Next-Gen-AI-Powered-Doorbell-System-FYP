import { useState, useEffect, useCallback, useMemo } from "react";
import axios from "axios";
import LoginPage from './Login.jsx'; // We need the separate Login page component

import {
  Typography, Box, Button, CircularProgress, Paper, TextField,
  Alert, CssBaseline, Grid, Chip, Drawer, List, ListItem, DialogActions,
  ListItemButton, ListItemIcon, ListItemText, Dialog, DialogTitle,
  DialogContent, IconButton, ToggleButtonGroup, ToggleButton, Toolbar
} from "@mui/material";
import { DataGrid, GridToolbarQuickFilter } from "@mui/x-data-grid";

// --- Icons ---
import DashboardIcon from '@mui/icons-material/Dashboard';
import SendIcon from '@mui/icons-material/Send';
import MailOutlineIcon from '@mui/icons-material/MailOutline';
import HistoryIcon from '@mui/icons-material/History';
import PermMediaIcon from '@mui/icons-material/PermMedia';
import RefreshIcon from "@mui/icons-material/Refresh";
import CloseIcon from '@mui/icons-material/Close';
import LogoutIcon from '@mui/icons-material/Logout';

// --- Configuration ---
axios.defaults.baseURL = "/api";
const DRAWER_WIDTH = 240;
const BACKEND_URL = "http://127.0.0.1:8001"; // Your backend's address



// --- Axios Interceptor ---
const setupAxiosInterceptors = (token) => {
  axios.interceptors.request.use(config => {
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  }, error => {
    return Promise.reject(error);
  });
};

function QuickFilterToolbar() {
  return (
    <Box sx={{ p: 1, pb: 0, borderBottom: '1px solid', borderColor: 'divider' }}>
      <GridToolbarQuickFilter />
    </Box>
  );
}

// =================================================================================
// The Main Dashboard Component (Fully Restored)
// =================================================================================
function Dashboard({ onLogout }) {
  // --- State Declarations ---
  const [activePage, setActivePage] = useState("Dashboard");
  const [loading, setLoading] = useState(false);

  /** RFC-5322-lite email validator.  Good enough for UI. */
  const isValidEmail = (e) =>
  /^[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+$/i.test(e);

  const [invites, setInvites] = useState([]);
  const [registrationLogs, setRegistrationLogs] = useState([]);
  const [attendanceLogs, setAttendanceLogs] = useState([]);
  const [mediaData, setMediaData] = useState({ proofs: [], archives: [] });
  const [dashboardStats, setDashboardStats] = useState(null);

  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteStatus, setInviteStatus] = useState({ msg: "", type: "" });

  const [mediaFilter, setMediaFilter] = useState('All');
  const [selectedUser, setSelectedUser] = useState(null);

  const [regDetailsOpen, setRegDetailsOpen] = useState(false);
  const [mediaDetailsOpen, setMediaDetailsOpen] = useState(false);
  const [selectedRowData, setSelectedRowData] = useState(null);

  const [imageDialogOpen, setImageDialogOpen] = useState(false);
  const [imageUrl, setImageUrl] = useState('');

  const [confirmOpen, setConfirmOpen]   = useState(false);
  const openConfirm  = () => setConfirmOpen(true);
  const closeConfirm = () => setConfirmOpen(false);
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
  const openConfirmDelete  = () => setConfirmDeleteOpen(true);
  const closeConfirmDelete = () => setConfirmDeleteOpen(false);
  const handleConfirmedDelete = () => {
    deleteRegistration(selectedRowData.UserID);
    closeConfirmDelete();
  };
  const [confirmApproveOpen, setConfirmApproveOpen] = useState(false);
  const openConfirmApprove  = () => setConfirmApproveOpen(true);
  const closeConfirmApprove = () => setConfirmApproveOpen(false);
  const handleConfirmedApprove = () => {
    approveRegistration(selectedRowData.UserID);
    closeConfirmApprove();
  };

  // --- Data Fetchers ---
  const fetchInvites = useCallback(async () => { setLoading(true); try { const { data } = await axios.get("/invites"); setInvites(data); } catch (e) { console.error("Failed to fetch invites:", e); } setLoading(false); }, []);
  const fetchRegistrationLogs = useCallback(async () => { setLoading(true); try { const { data } = await axios.get("/registration-logs"); setRegistrationLogs(data); } catch (e) { console.error("Failed to fetch registration logs:", e); } setLoading(false); }, []);
  const fetchAttendanceLogs = useCallback(async () => { setLoading(true); try { const { data } = await axios.get("/attendance-logs"); setAttendanceLogs(data); } catch (e) { console.error("Failed to fetch attendance logs:", e); } setLoading(false); }, []);
  const fetchMedia = useCallback(async () => { setLoading(true); try { const { data } = await axios.get("/media"); setMediaData(data); } catch (e) { console.error("Failed to fetch media:", e); } setLoading(false); }, []);
  const fetchDashboardStats = useCallback(async () => { setLoading(true); try { const { data } = await axios.get("/dashboard-stats"); setDashboardStats(data); } catch (e) { console.error("Failed to fetch dashboard stats:", e); } setLoading(false); }, []);

  // --- Handlers ---
  const handleOpenRegDetails = (row) => { setSelectedRowData(row); setRegDetailsOpen(true); };
  const handleCloseRegDetails = () => { setRegDetailsOpen(false); };
  const handleOpenMediaDetails = (uid) => { setSelectedUser(uid); setMediaDetailsOpen(true); };
  const handleCloseMediaDetails = () => { setMediaDetailsOpen(false); };
  const handleViewAttendanceImage = (row) => {
    if (row.proofphotopath && row.proofphotopath !== 'N/A') {
      const subPath = row.proofphotopath.replace(/^backend\//, ""); // strip leading “backend/”
      const fullUrl = `${BACKEND_URL}/api/media/raw/${subPath}`
      setImageUrl(fullUrl);
      setImageDialogOpen(true);
    }
  };
  const handleViewUserMedia = (uid) => {
    handleCloseRegDetails();
    setActivePage('Media');
    setTimeout(() => { handleOpenMediaDetails(uid); }, 100);
  };

  const sendInvite = async () => {
    const email = inviteEmail.trim().toLowerCase();
    if (!isValidEmail(email)) { setInviteStatus({ msg: "Invalid email format.", type: "error" }); return; }
    setLoading(true);
    setInviteStatus({ msg: "", type: "" });
    try {
      await axios.post(`/invite?email=${encodeURIComponent(email)}`);
      setInviteStatus({ msg: `Invite sent successfully to ${inviteEmail}!`, type: "success" });
      setInviteEmail("");
      fetchInvites();
    } catch (error) {
      const errorMsg = error.response?.data?.detail || "An unexpected error occurred.";
      setInviteStatus({ msg: `Failed to send invite: ${errorMsg}`, type: "error" });
    }
    setLoading(false);
  };

  const handleInviteClick = () => {
    const cleaned = inviteEmail.trim().toLowerCase();
    if (!cleaned) { setInviteStatus({ msg: "Email cannot be empty.", type: "error" }); return; }
    if (!isValidEmail(cleaned)) { setInviteStatus({ msg: "Invalid email format.", type: "error" }); return; }
    setInviteEmail(cleaned);
    openConfirm();
  };

  const approveRegistration = async (uid) => {
    try {
      // note the `?uid=` here
      await axios.post(`/registration/${uid}/approve`);
      fetchRegistrationLogs();
      setSelectedRowData(prev => ({ ...prev, ManualApproval: 'True' }));
      setInviteStatus({ msg: 'User approved successfully', type: 'success' });
    } catch (e) {
      setInviteStatus({ msg: 'Approve failed: ' + (e.response?.data || e.message), type: 'error' });
    }
  };

  const deleteRegistration = async (uid) => {
    try {
      await axios.delete(`/registration/${uid}`);
      fetchRegistrationLogs();
      fetchMedia();
      handleCloseRegDetails();
    } catch (e) {
      setInviteStatus({ msg: 'Delete failed: ' + (e.response?.data || e.message), type: 'error' });
    }
  };

  const handleToggleUserStatus = async () => {
    if (!selectedRowData) return;
    const { UserID, status } = selectedRowData;
    const newStatus = status === 'blocked' ? 'allowed' : 'blocked';

    try {
      await axios.post(`/users/${UserID}/status`, { status: newStatus });
      setSelectedRowData(prev => ({ ...prev, status: newStatus }));
      fetchRegistrationLogs();
      setInviteStatus({ msg: `User status changed to ${newStatus}`, type: 'success' });
    } catch(e) {
      setInviteStatus({ msg: 'Status update failed: ' + (e.response?.data?.detail || e.message), type: 'error' });
    }
  };

  const isOverallPass = (selectedRowData?.LivePass === "True" && selectedRowData?.DeepfakePass === "True") || selectedRowData?.ManualApproval === "True";

  useEffect(() => {
    if (activePage === "Dashboard") fetchDashboardStats();
    if (activePage === "Invites") fetchInvites();
    if (activePage === "Registration Logs") fetchRegistrationLogs();
    if (activePage === "Attendance Logs") fetchAttendanceLogs();
    if (activePage === "Media") fetchMedia();
  }, [activePage, fetchDashboardStats, fetchInvites, fetchRegistrationLogs, fetchAttendanceLogs, fetchMedia]);

  const filteredUserRows = useMemo(() => mediaData.archives.sort((a,b)=> new Date(b.date) - new Date(a.date)).map((rec, idx) => ({ id: `${rec.uid}-${idx}`, ...rec })).filter(row => mediaFilter === 'All' || row.status === mediaFilter), [mediaData.archives, mediaFilter]);
  const userMediaDetails = useMemo(() => {
    if (!selectedUser) return null;
    const cleanSelectedUser = selectedUser.trim();
    const proofFilename = mediaData.proofs.find(
      (p) => p.split("/").pop().startsWith(cleanSelectedUser)   // look at basename
    );

    return {
      proof: proofFilename ? `${BACKEND_URL}/api/media/raw/${proofFilename}` : "",
      archives: mediaData.archives.filter(a => a.uid.trim() === cleanSelectedUser)
    };
  }, [selectedUser, mediaData.proofs, mediaData.archives]);

  const navItems = [{ text: 'Dashboard', icon: <DashboardIcon /> }, { text: 'Invites', icon: <MailOutlineIcon /> }, { text: 'Registration Logs', icon: <HistoryIcon /> }, { text: 'Attendance Logs', icon: <HistoryIcon /> }, { text: 'Media', icon: <PermMediaIcon /> }];

  const attendanceColumns = [
    { field: 'log_type', headerName: 'Event Type', width: 160, renderCell: (params) => {
        const type = params.value;
        let label, color;
        switch (type) {
          case 'RECOGNITION': label = 'Recognition'; color = 'success'; break;
          case 'LIVENESS_ALERT': label = 'Liveness Alert'; color = 'warning'; break;
          case 'DEEPFAKE_ALERT': label = 'Deepfake Alert'; color = 'error'; break;
          case 'BLOCKLIST_ALERT': label = 'Blocklist Alert'; color = 'error'; break;
          default: label = type; color = 'default';
        }
        return <Chip label={label} color={color} size="small" variant="outlined" />;
      },
    },
    { field: "userid", headerName: "User ID", flex: 1, minWidth: 150 }, { field: "name", headerName: "Name", flex: 1, minWidth: 120 },
    { field: "confidence", headerName: "Confidence", flex: 0.5, minWidth: 100 }, { field: "timestamp", headerName: "Timestamp", flex: 1, type: 'dateTime', valueGetter: v => v ? new Date(v) : null, minWidth: 170 },
    { field: "actions", headerName: "Details", sortable: false, filterable: false, disableColumnMenu: true, width: 120, renderCell: (params) => ( <Button variant="outlined" size="small" disabled={!params.row.proofphotopath || params.row.proofphotopath === 'N/A'} onClick={() => handleViewAttendanceImage(params.row)}>View Image</Button> ),},
  ];

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <Drawer variant="permanent" sx={{ width: DRAWER_WIDTH, flexShrink: 0, '& .MuiDrawer-paper': { width: DRAWER_WIDTH, boxSizing: 'border-box' }}}>
        <Toolbar><Typography variant="h6" noWrap>Admin Panel</Typography></Toolbar>
        <Box sx={{ overflow: 'auto', flexGrow: 1 }}><List>{navItems.map((item) => (<ListItem key={item.text} disablePadding><ListItemButton selected={activePage === item.text} onClick={() => setActivePage(item.text)}><ListItemIcon>{item.icon}</ListItemIcon><ListItemText primary={item.text} /></ListItemButton></ListItem>))}</List></Box>
        <Button onClick={onLogout} variant="contained" color="error" startIcon={<LogoutIcon />} sx={{ m: 2 }}>Logout</Button>
      </Drawer>

      <Box component="main" sx={{ flexGrow: 1, bgcolor: 'grey.100', p: 3, height: '100vh', overflow: 'auto' }}>
        <Toolbar />

        {/* --- ALL PAGE CONTENT RESTORED BELOW --- */}

        {activePage === 'Dashboard' && (
          <Box><Grid container spacing={3}><Grid item xs={12} sm={4}><Paper sx={{ p: 2, textAlign: 'center' }}><Typography color="text.secondary">Total Registered Users</Typography><Typography variant="h4" fontWeight="bold">{loading ? <CircularProgress size={24}/> : dashboardStats?.totalUsers}</Typography></Paper></Grid><Grid item xs={12} sm={4}><Paper sx={{ p: 2, textAlign: 'center' }}><Typography color="text.secondary">Attendance Today</Typography><Typography variant="h4" fontWeight="bold">{loading ? <CircularProgress size={24}/> : dashboardStats?.attendanceToday}</Typography></Paper></Grid><Grid item xs={12} sm={4}><Paper sx={{ p: 2, textAlign: 'center' }}><Typography color="text.secondary">Total Invites Sent</Typography><Typography variant="h4" fontWeight="bold">{loading ? <CircularProgress size={24}/> : dashboardStats?.totalInvites}</Typography></Paper></Grid><Grid item xs={12}><Paper sx={{ p: 2 }}><Typography variant="h6" gutterBottom>Recent Attendance</Typography><List>{dashboardStats?.recentEvents?.filter(event => event.log_type === 'RECOGNITION').length > 0 ? (dashboardStats.recentEvents.filter(event => event.log_type === 'RECOGNITION').slice(0).reverse().map((event, index) => (<ListItem key={index} divider><ListItemText primary={event.name} secondary={`Confidence: ${event.confidence}`}/><Typography variant="body2" color="text.secondary">{new Date(event.timestamp).toLocaleString()}</Typography></ListItem>))) : (<Typography sx={{ p: 2, textAlign: 'center' }} color="text.secondary">No recent events found.</Typography>)}</List></Paper></Grid></Grid></Box>
        )}

        {activePage === 'Invites' && (
          <Box><Paper elevation={2} sx={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 120px)' }}><Box sx={{ p: 2 }}><Typography variant="h6" gutterBottom>Invite New User</Typography><Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}><TextField label="User Email" variant="outlined" size="small" value={inviteEmail} onChange={(e) => setInviteEmail(e.target.value)} disabled={loading} sx={{ flexGrow: 1, minWidth: '250px' }} /><Button variant="contained" startIcon={<SendIcon />} onClick={handleInviteClick} disabled={loading}>Send Invite</Button><Button variant="outlined" startIcon={<RefreshIcon />} onClick={fetchInvites} disabled={loading}>Refresh List</Button>{loading && <CircularProgress size={24} />}</Box>{inviteStatus.msg && (<Alert severity={inviteStatus.type} sx={{ mt: 2 }}>{inviteStatus.msg}</Alert>)}</Box><Box sx={{ borderBottom: 1, borderColor: 'divider' }} /><DataGrid rows={invites.map((i) => ({ id: i.token, ...i }))} columns={[{ field: "token", headerName: "Token", flex: 2, minWidth: 200 }, { field: "email", headerName: "Email", flex: 1, minWidth: 150 }, { field: "used", headerName: "Used", type: 'boolean', width: 80 }, { field: "issued", headerName: "Issued", flex: 1, type: 'dateTime', valueGetter: v => v ? new Date(typeof v === "number" && v < 2e11 ? v * 1000 : v) : null, minWidth: 170 ,}]}  initialState={{ sorting: { sortModel: [{ field: "issued", sort: "desc" }] } }}slots={{ toolbar: QuickFilterToolbar }} disableRowSelectionOnClick sx={{ flexGrow: 1, border: 0 }} /></Paper></Box>
        )}

        {activePage === "Registration Logs" && (
          <Box><Paper elevation={2}><Box sx={{ p: 2 }}><Typography variant="h5" gutterBottom>Registration Log</Typography>{inviteStatus.msg && (<Alert severity={inviteStatus.type} sx={{ mb: 2 }}>{inviteStatus.msg}</Alert>)}</Box><Box sx={{ height: "calc(100vh - 220px)", width: "100%" }}><DataGrid loading={loading} rows={registrationLogs.map((r) => ({ ...r, id: r.UserID }))} columns={[{ field: "UserID", headerName: "User ID", flex: 1, minWidth: 150 },{ field: "Name",   headerName: "Name",    flex: 1, minWidth: 120 }, { field: "status", headerName: "Status", width: 120, renderCell: (params) => <Chip label={params.value} color={params.value === 'blocked' ? 'error' : 'default'} size="small"/>}, { field: "overallStatus", headerName: "Overall Status", width: 160, renderCell: (params) => { const r = params.row; const systemPass = r.LivePass === "True" && r.DeepfakePass === "True"; const manual = r.ManualApproval === "True"; let label, color; if (manual) { label = "Manual"; color = "warning"; } else if (systemPass) { label = "Pass"; color = "success"; } else { label = "Fail"; color = "error"; } return (<Chip label={label} color={color} size="small" variant="outlined" />); },},{ field: "Email", headerName: "Email", flex: 1, minWidth: 180 },{ field: "Timestamp", headerName: "Timestamp", flex: 1, type: "dateTime", valueGetter: (v) => new Date(v), minWidth: 170,},{ field: "actions", headerName: "Details", sortable: false, filterable: false, disableColumnMenu: true, width: 150, renderCell: (params) => (<Button variant="outlined" size="small" onClick={() => handleOpenRegDetails(params.row)} >More Details</Button>),},]} initialState={{ sorting: { sortModel: [{ field: "Timestamp", sort: "desc" }] } }} slots={{ toolbar: QuickFilterToolbar }} disableRowSelectionOnClick sx={{ border: 0 }} /></Box></Paper></Box>
        )}

        {activePage === 'Attendance Logs' && (
          <Box><Paper elevation={2}><Box sx={{ p: 2 }}><Typography variant="h5" gutterBottom>Attendance Log</Typography></Box><Box sx={{ height: 'calc(100vh - 180px)', width: '100%' }}><DataGrid loading={loading} rows={attendanceLogs.map((r, index) => ({ ...r, id: index }))} columns={attendanceColumns} slots={{ toolbar: QuickFilterToolbar }} initialState={{ sorting: { sortModel: [{ field: 'timestamp', sort: 'desc' }] } }}/></Box></Paper></Box>
        )}

        {activePage === 'Media' && (
          <Box><Paper elevation={2}><Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid', borderColor: 'divider' }}><Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}><Typography variant="body1" fontWeight="bold">Status:</Typography><ToggleButtonGroup color="primary" value={mediaFilter} exclusive onChange={(e, newFilter) => { if (newFilter !== null) setMediaFilter(newFilter); }} size="small"><ToggleButton value="All">All</ToggleButton><ToggleButton value="PASS">Pass</ToggleButton><ToggleButton value="FAIL">Fail</ToggleButton><ToggleButton value="MANUAL">Manual</ToggleButton></ToggleButtonGroup></Box><Button variant="outlined" startIcon={<RefreshIcon />} onClick={fetchMedia} disabled={loading}>Refresh Media</Button></Box><Box sx={{ height: 'calc(100vh - 200px)', width: '100%' }}><DataGrid loading={loading} rows={filteredUserRows} columns={[{ field: "uid", headerName: "User ID", flex: 1 }, { field: "status", headerName: "Status", flex: 0.5 }, { field: "date", headerName: "Date", flex: 0.5 }]} onRowClick={(params) => handleOpenMediaDetails(params.row.uid)} sx={{ border: 0, '& .MuiDataGrid-row': { cursor: 'pointer' } }} /></Box></Paper></Box>
        )}
      </Box>

      {/* All Dialogs */}
      <Dialog open={regDetailsOpen} onClose={handleCloseRegDetails} fullWidth maxWidth="sm">
        <DialogTitle sx={{ pr: 5 }}>Registration Score Details<IconButton aria-label="close" onClick={handleCloseRegDetails} sx={{ position: 'absolute', right: 8, top: 8, color: (theme) => theme.palette.grey[500], }}><CloseIcon /></IconButton></DialogTitle>
        <DialogContent dividers>
          {selectedRowData && (
            <Box sx={{ p: 1, minWidth: 400 }}>

              {/* ─── Basic info ──────────────────────── */}
              <Typography variant="h6" gutterBottom>
                User: {selectedRowData.Name} ({selectedRowData.UserID})
              </Typography>

              <Typography component="div" sx={{ mb: 2 }}>
                Status:&nbsp;
                <Chip
                  label={selectedRowData.status}
                  color={selectedRowData.status === "blocked" ? "error" : "success"}
                  size="small"
                />
              </Typography>

              {/* ─── Liveness scores ─────────────────── */}
              <Typography variant="subtitle1" sx={{ fontWeight: "bold" }}>
                Liveness&nbsp;
                {selectedRowData.LivePass === "True" ? "✔" : "✖"}
              </Typography>
              <Typography component="div" sx={{ ml: 2, mb: 1 }}>
                C:&nbsp;{selectedRowData.Spoof_Center}&nbsp;|&nbsp;
                L:&nbsp;{selectedRowData.Spoof_Left}&nbsp;|&nbsp;
                R:&nbsp;{selectedRowData.Spoof_Right}
              </Typography>

              {/* ─── Deep‑fake scores ────────────────── */}
              <Typography variant="subtitle1" sx={{ fontWeight: "bold" }}>
                Deepfake&nbsp;
                {selectedRowData.DeepfakePass === "True" ? "✔" : "✖"}
              </Typography>
              <Typography component="div" sx={{ ml: 2 }}>
                C:&nbsp;{selectedRowData.DF_Center}&nbsp;|&nbsp;
                L:&nbsp;{selectedRowData.DF_Left}&nbsp;|&nbsp;
                R:&nbsp;{selectedRowData.DF_Right}
              </Typography>

            </Box>
          )}
        </DialogContent>

        <DialogActions sx={{ p: '16px 24px', display: 'flex', justifyContent: 'space-between', width: '100%', }}>
          <Button variant="contained" color="error" onClick={openConfirmDelete}>Delete User</Button>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button variant="contained" color={selectedRowData?.status === 'blocked' ? 'success' : 'warning'} onClick={handleToggleUserStatus}>
              {selectedRowData?.status === 'blocked' ? 'Whitelist User' : 'Block User'}
            </Button>
            <Button variant="contained" color="success" disabled={isOverallPass} onClick={openConfirmApprove}>Approve</Button>
            <Button variant="outlined" onClick={() => handleViewUserMedia(selectedRowData.UserID)}>View Media</Button>
          </Box>
        </DialogActions>
      </Dialog>
      <Dialog open={imageDialogOpen} onClose={() => setImageDialogOpen(false)} maxWidth="md"><DialogTitle>View Image<IconButton aria-label="close" onClick={() => setImageDialogOpen(false)} sx={{ position: 'absolute', right: 8, top: 8, color: (theme) => theme.palette.grey[500] }}><CloseIcon /></IconButton></DialogTitle><DialogContent dividers><img src={imageUrl} alt="Attendance log detail" style={{ width: '100%', height: 'auto' }} /></DialogContent><DialogActions><Button onClick={() => setImageDialogOpen(false)}>Close</Button></DialogActions></Dialog>
      <Dialog open={confirmOpen} onClose={closeConfirm}><DialogTitle>Confirm Invitation</DialogTitle><DialogContent dividers><Typography sx={{ mb: 1 }}>Send invite to <strong>{inviteEmail}</strong>?</Typography><Typography variant="body2" color="text.secondary">Please double-check that the email address is correct.</Typography></DialogContent><DialogActions><Button onClick={closeConfirm}>Cancel</Button><Button variant="contained" startIcon={<SendIcon />} onClick={() => { closeConfirm(); sendInvite(); }}>Confirm &amp; Send</Button></DialogActions></Dialog>
      <Dialog open={confirmDeleteOpen} onClose={closeConfirmDelete}><DialogTitle>Confirm Delete</DialogTitle><DialogContent dividers><Typography>Permanently remove <strong>{selectedRowData?.Name}</strong> ({selectedRowData?.UserID})?</Typography><Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>This action cannot be undone.</Typography></DialogContent><DialogActions><Button onClick={closeConfirmDelete}>Cancel</Button><Button variant="contained" color="error" onClick={handleConfirmedDelete}>Yes, Delete</Button></DialogActions></Dialog>
      <Dialog open={confirmApproveOpen} onClose={closeConfirmApprove}><DialogTitle>Confirm Manual Approval</DialogTitle><DialogContent dividers><Typography sx={{ mb: 1 }}>Approve <strong>{selectedRowData?.Name}</strong> ({selectedRowData?.UserID})?</Typography><Typography variant="body2" color="text.secondary">Their face embedding will be saved permanently.</Typography></DialogContent><DialogActions><Button onClick={closeConfirmApprove}>Cancel</Button><Button variant="contained" color="success" onClick={handleConfirmedApprove}>Yes, Approve</Button></DialogActions></Dialog>
      <Dialog open={mediaDetailsOpen} onClose={handleCloseMediaDetails} fullWidth maxWidth="md"><DialogTitle>User Media Details</DialogTitle><IconButton aria-label="close" onClick={handleCloseMediaDetails} sx={{ position: 'absolute', right: 8, top: 8, color: (theme) => theme.palette.grey[500] }}><CloseIcon /></IconButton><DialogContent dividers>{userMediaDetails && (<Grid container spacing={2}><Grid item xs={12} md={4}><Typography variant="h6" gutterBottom>Proof Photo</Typography>{userMediaDetails.proof ? <img src={userMediaDetails.proof} alt={`Proof for ${selectedUser}`} style={{ width: '100%', borderRadius: 8 }}/> : <Typography>No proof photo.</Typography>}</Grid><Grid item xs={12} md={8}><Typography variant="h6" gutterBottom>Archived Videos</Typography><Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>{userMediaDetails.archives.length > 0 ? userMediaDetails.archives.map((archive, i) => (<Box key={i}><Typography variant="subtitle2">{archive.date} - {archive.status}</Typography>{archive.files.map((file, j) => (<video key={j} src={`${BACKEND_URL}/api/media/raw/archive/${archive.date}/${archive.status}_${archive.uid}/${file}`} controls style={{ width: '100%', borderRadius: 8, marginTop: 4 }}/>))}</Box>)) : <Typography>No archived videos.</Typography>}</Box></Grid></Grid>)}</DialogContent><Button onClick={handleCloseMediaDetails} sx={{ m: 1 }}>Close</Button></Dialog>
    </Box>
  );
}

// =================================================================================
// The Main App Component (now only handles login logic)
// =================================================================================
export default function App() {
  const [token, setToken] = useState(localStorage.getItem('admin_token'));

  // as soon as we know we have a token, install the interceptor
  if (token) {
    setupAxiosInterceptors(token);
  }

  const handleLoginSuccess = (newToken) => {
    localStorage.setItem('admin_token', newToken);
    setToken(newToken);
  };

  const handleLogout = () => {
    localStorage.removeItem('admin_token');
    setToken(null);
  };

  if (!token) {
    return <LoginPage onLoginSuccess={handleLoginSuccess} />;
  }

  return <Dashboard onLogout={handleLogout} />;
}